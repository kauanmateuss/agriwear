import os
import numpy as np
import json
import time
from PIL import Image

# Importa as bibliotecas específicas do RPi
try:
    import tflite_runtime.interpreter as tflite
    import RPi.GPIO as GPIO
    from LoraModule import SX1276_Corrected
except ImportError:
    print(
        "Erro: Este script foi feito para rodar no Raspberry Pi com as bibliotecas tflite_runtime, RPi.GPIO e LoraModule.")
    print("Para testar em um notebook, instale o 'tensorflow' completo e comente as importações de GPIO/Lora.")
    # Tenta usar o tensorflow completo como alternativa
    import tensorflow.lite as tflite

# --- Configuração GPIO ---
BUTTON_PIN = 3
LED_AZUL = 21
LED_AMARELO = 5

# --- Configuração do Modelo ---
MODEL_PATH = "best_int8.tflite"  # Seu modelo INT8 [cite: 1, cell 19]
IMAGE_PATH = "frame.jpg"  # Nome do arquivo da foto capturada
CLASS_NAMES = ["cacho"]
CONF_THRESH = 0.3  # Limite de confiança
NMS_THRESH = 0.4  # Limite de sobreposição


# --- Função NMS (Puro NumPy) ---
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


# --- Função Principal ---
def main():
    # --- 1. Setup do Hardware ---
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED_AMARELO, GPIO.OUT)
    GPIO.setup(LED_AZUL, GPIO.OUT)
    GPIO.output(LED_AMARELO, GPIO.HIGH)  # LED Amarelo = Pronto
    GPIO.output(LED_AZUL, GPIO.LOW)

    lora = None
    try:
        print("=== Conectando LoRa ===")
        lora = SX1276_Corrected(
            spi_bus=0, spi_device=0, reset_pin=25,
            cs_pin=8, dio0_pin=24, frequency=915000000, power=14
        )
        print("✅ LoRa conectado.")
    except Exception as e:
        print(f"❌ Erro ao conectar LoRa: {e}")
        print("Continuando sem LoRa...")

    # --- 2. Setup do Modelo TFLite ---
    print(f"Carregando modelo {MODEL_PATH}...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_h, input_w = input_details[0]["shape"][1:3]
    input_dtype = input_details[0]['dtype']

    # Verifica se o modelo é INT8 (essencial para o processamento correto)
    is_int8 = input_dtype == np.int8
    if is_int8:
        scale, zero_point = input_details[0]['quantization']
        print(f"✅ Modelo INT8 carregado. Shape: [{input_h}, {input_w}]")
    else:
        print(f"⚠️ Aviso: Modelo FLOAT32 carregado. Shape: [{input_h}, {input_w}]")
        print("O desempenho no RPi 2B será muito lento.")

    print("\n=== Contagem de cachos - Raspberry Pi 2B ===")
    print("Pressione o botão para capturar e analisar, ou Ctrl+C para sair.")

    # --- 3. Loop Principal ---
    try:
        while True:
            # Espera o botão ser pressionado
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                GPIO.output(LED_AMARELO, GPIO.LOW)  # LED Amarelo = Ocupado
                GPIO.output(LED_AZUL, GPIO.HIGH)  # LED Azul = Processando

                print("\n" + "=" * 50)
                print("📸 Capturando foto...")
                # 1. CAPTURA DE IMAGEM (do seu script [cite: 1, user_code])
                os.system(f"rpicam-jpeg --output {IMAGE_PATH} -t 500 --width 640 --height 480")

                if not os.path.exists(IMAGE_PATH):
                    print("❌ Erro: Foto não foi capturada.")
                    time.sleep(1)
                    continue

                print(f"✅ Foto capturada: {IMAGE_PATH}")

                # 2. PRÉ-PROCESSAMENTO (Com Pillow, sem OpenCV)
                img_pil = Image.open(IMAGE_PATH).convert('RGB')
                img_resized = img_pil.resize((input_w, input_h), Image.BILINEAR)
                img_resized_norm = np.array(img_resized) / 255.0

                # 3. PREPARAR TENSOR (Com quantização INT8)
                if is_int8:
                    input_tensor = (img_resized_norm / scale + zero_point).astype(np.int8)
                else:
                    input_tensor = img_resized_norm.astype(np.float32)
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # 4. INFERÊNCIA
                print("🧠 Analisando imagem...")
                start_infer = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])[0]
                infer_time = time.time() - start_infer
                print(f"⏱️ Tempo de inferência: {infer_time:.2f} segundos")

                # 5. PÓS-PROCESSAMENTO (Com NumPy, sem OpenCV) [cite: 1, user_code]
                boxes_yolo_norm = preds[:4, :].T
                confs = preds[4, :]
                boxes_xyxy = np.zeros_like(boxes_yolo_norm)
                boxes_xyxy[:, 0] = (boxes_yolo_norm[:, 0] - boxes_yolo_norm[:, 2] / 2)
                boxes_xyxy[:, 1] = (boxes_yolo_norm[:, 1] - boxes_yolo_norm[:, 3] / 2)
                boxes_xyxy[:, 2] = (boxes_yolo_norm[:, 0] + boxes_yolo_norm[:, 2] / 2)
                boxes_xyxy[:, 3] = (boxes_yolo_norm[:, 1] + boxes_yolo_norm[:, 3] / 2)

                mask = confs > CONF_THRESH
                boxes_filtered = boxes_xyxy[mask]
                confs_filtered = confs[mask]

                indices = non_max_suppression(boxes_filtered, confs_filtered, NMS_THRESH)
                detection_count = len(indices)

                print(f"🍇 RESULTADO: {detection_count} 'cachos' detectados.")

                # 6. ENVIO LORA (do seu script [cite: 1, user_code])
                if lora:
                    payload = {"frutos": detection_count}
                    message = json.dumps(payload)
                    print(f"📡 Enviando via LoRa: {message}")
                    success = lora.send_message(message.encode())

                    if success:
                        print("✅ Mensagem enviada com sucesso!")
                    else:
                        print("❌ Falha no envio LoRa.")

                GPIO.output(LED_AMARELO, GPIO.HIGH)  # LED Amarelo = Pronto
                GPIO.output(LED_AZUL, GPIO.LOW)

                # Limpa a foto
                os.remove(IMAGE_PATH)

                # Espera para evitar cliques duplicados
                time.sleep(0.5)

            time.sleep(0.05)  # Loop de espera do botão

    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        # Limpa os pinos GPIO ao sair
        GPIO.cleanup()
        print("GPIO limpo. Tchau!")


# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()