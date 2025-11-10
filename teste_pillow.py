import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont
import json
from LoraModule import SX1276_Corrected
import RPi.GPIO as GPIO

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

#Conexoes gpio
BUTTON_PIN = 3
LED_AZUL = 7
LED_AMARELO = 15
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
GPIO.setup(LED_AMARELO, GPIO.OUT)
GPIO.setup(LED_AZUL, GPIO.OUT)
GPIO.output(LED_AMARELO, GPIO.HIGH)

# Conexao do modulo Lora

try:
    print("=== Conectando Lora ===")
    
    # Inicializar LoRa
    lora = SX1276_Corrected(
        spi_bus=0,
        spi_device=0,
        reset_pin=25,
        cs_pin=8,
        dio0_pin=24,
        frequency=915000000,
        power=14
    )
except Exception as e:
    lora = None
    print(f"❌ Erro: {e}")

# --- Configuração ---
MODEL_PATH = "grad_final_int8.tflite"  # Seu modelo INT8
IMAGE_PATH = "frame.jpeg"  # A imagem que você quer contar
OUTPUT_PATH = "resultado_pillow.jpg"
CLASS_NAMES = ["cacho"]
CONF_THRESH = 0.7
NMS_THRESH = 0.4


# --------------------

# (Função NMS com NumPy - a mesma de antes)
def non_max_suppression(boxes_xyxy, scores, iou_threshold):
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep_indices


# --- Script Principal ---
print(f"Carregando modelo {MODEL_PATH}...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]["shape"][1:3]
input_dtype = input_details[0]['dtype']

while True:
    GPIO.output(LED_AMARELO, GPIO.HIGH)
    GPIO.output(LED_AZUL, GPIO.LOW)
    print("Aperte o botao para capturar a imagem")
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            break
    GPIO.output(LED_AMARELO, GPIO.LOW)
    GPIO.output(LED_AZUL, GPIO.HIGH)
    # Captura foto com Pi Camera
    print("📸 Capturando foto...")
    os.system("rpicam-jpeg --output frame.jpg -t 1")
    #os.system("jpegtran -rotate 90 -outfile frame.jpg frame.jpg")

    # 1. Carregar e Pré-processar Imagem (com Pillow)
    print(f"Lendo imagem {IMAGE_PATH}...")
    try:
        img = Image.open("frame.jpg").convert('RGB')
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em {IMAGE_PATH}")
        exit()
    
    orig_w, orig_h = img.size
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    img_resized_norm = np.array(img_resized) / 255.0  # Normaliza para [0, 1]
    
    # 2. Preparar Tensor de Entrada
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_tensor = (img_resized_norm / scale + zero_point).astype(np.int8)
    else:
        input_tensor = img_resized_norm.astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # 3. Rodar Inferência
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # 4. Pós-processamento e NMS (NumPy)
    boxes_yolo_norm = preds[:4, :].T
    confs = preds[4, :]
    boxes_xyxy = np.zeros_like(boxes_yolo_norm)
    boxes_xyxy[:, 0] = (boxes_yolo_norm[:, 0] - boxes_yolo_norm[:, 2] / 2) * orig_w  # x1
    boxes_xyxy[:, 1] = (boxes_yolo_norm[:, 1] - boxes_yolo_norm[:, 3] / 2) * orig_h  # y1
    boxes_xyxy[:, 2] = (boxes_yolo_norm[:, 0] + boxes_yolo_norm[:, 2] / 2) * orig_w  # x2
    boxes_xyxy[:, 3] = (boxes_yolo_norm[:, 1] + boxes_yolo_norm[:, 3] / 2) * orig_h  # y2
    
    mask = confs > CONF_THRESH
    boxes_filtered = boxes_xyxy[mask]
    confs_filtered = confs[mask]
    indices = non_max_suppression(boxes_filtered, confs_filtered, NMS_THRESH)
    
    # 5. Desenhar caixas (com Pillow)
    draw = ImageDraw.Draw(img)
    try:
        # Tenta carregar uma fonte melhor, se falhar, usa a padrão
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for i in indices:
        x1, y1, x2, y2 = boxes_filtered[i]
        conf = confs_filtered[i]
        label = f"{CLASS_NAMES[0]}: {conf:.2f}"
    
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 15), label, fill="green", font=font)
    
    # 6. Salvar
    img.save(OUTPUT_PATH)
    print(f"--- Detecção concluída! {len(indices)} 'cachos' encontrados. ---")
    print(f"Imagem salva em {OUTPUT_PATH}")
    payload = {"frutos": len(indices)}
    message = json.dumps(payload)
    if lora:
        success = lora.send_message(message.encode())
        
        if success:
            print("Mensagem enviada com sucesso!")
        else:
            print("❌ Falha no envio, tentando novamente...")
    else:
        print("Mensagem não enviada pois o lora n foi inicializado")

