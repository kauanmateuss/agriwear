#!/usr/bin/env python3

import spidev
import RPi.GPIO as GPIO
import time
import json

class SX1276_Corrected:
    def __init__(self, spi_bus=0, spi_device=0, reset_pin=25, cs_pin=8, dio0_pin=24, frequency=915000000, power=14):
        # Configuração GPIO
        self.reset_pin = reset_pin
        self.cs_pin = cs_pin
        self.dio0_pin = dio0_pin
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.reset_pin, GPIO.OUT)
        GPIO.setup(self.cs_pin, GPIO.OUT)
        GPIO.setup(self.dio0_pin, GPIO.IN)
        
        # Inicializar SPI
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0
        
        self.frequency = frequency
        self.power = power
        
        print("Inicializando módulo LoRa...")
        self.reset()
        self.init_lora()
        
    def reset(self):
        """Reset hardware do módulo"""
        GPIO.output(self.reset_pin, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(self.reset_pin, GPIO.HIGH)
        time.sleep(0.1)
    
    def write_register(self, address, value):
        """Escrever em um registrador"""
        GPIO.output(self.cs_pin, GPIO.LOW)
        self.spi.xfer2([address | 0x80, value])
        GPIO.output(self.cs_pin, GPIO.HIGH)
        time.sleep(0.001)  # Pequena pausa entre escritas
    
    def read_register(self, address):
        """Ler de um registrador"""
        GPIO.output(self.cs_pin, GPIO.LOW)
        response = self.spi.xfer2([address & 0x7F, 0x00])
        GPIO.output(self.cs_pin, GPIO.HIGH)
        return response[1]
    
    def init_lora(self):
        """Inicialização correta baseada no debug"""
        try:
            # Verificar versão do chip
            version = self.read_register(0x42)
            print(f"Versão do SX1276: 0x{version:02X}")
            
            if version != 0x12:
                print("Aviso: Versão do chip diferente do esperado")
            
            # 1. Colocar em modo Sleep
            self.write_register(0x01, 0x00)
            time.sleep(0.1)
            
            # 2. Configurar para modo LoRa (0x80)
            self.write_register(0x01, 0x80)
            time.sleep(0.1)
            
            # Verificar se modo LoRa foi ativado
            op_mode = self.read_register(0x01)
            print(f"Modo de operação: 0x{op_mode:02X}")
            
            if op_mode != 0x80:
                print("ERRO: Não foi possível ativar modo LoRa")
                return False
            
            # 3. Configurar frequência (915 MHz)
            frf = int(self.frequency * 524288 / 32000000)
            self.write_register(0x06, (frf >> 16) & 0xFF)  # FRF_MSB
            self.write_register(0x07, (frf >> 8) & 0xFF)   # FRF_MID
            self.write_register(0x08, frf & 0xFF)          # FRF_LSB
            print(f"Frequência configurada: {self.frequency} Hz")
            
            # 4. Configurar potência de TX
            self.write_register(0x09, 0xFF)  # Max power
            print("Potência configurada: máximo")
            
            # 5. Configurar LNA
            self.write_register(0x0C, 0x23)  # LNA boost
            
            # 6. Configurações do modem (usando valores que funcionaram no debug)
            self.write_register(0x1D, 0x72)  # BW=125kHz, CR=4/5
            self.write_register(0x1E, 0x94)  # SF=9, CRC enabled
            self.write_register(0x26, 0x04)  # Modem config 3
            
            # 7. Preamble length
            self.write_register(0x20, 0x00)  # MSB
            self.write_register(0x21, 0x08)  # LSB (8 symbols)
            
            # 8. Sync word
            self.write_register(0x39, 0x12)
            
            # 9. Configurar FIFO
            self.write_register(0x0E, 0x00)  # Tx base address
            self.write_register(0x0F, 0x00)  # Rx base address
            
            # 10. Colocar em modo standby
            self.write_register(0x01, 0x81)
            time.sleep(0.1)
            
            print("✅ LoRa inicializado com sucesso!")
            return True
            
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            return False
    
    def send_message(self, message):
        """Enviar mensagem via LoRa - Versão Corrigida"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        print(f"Preparando para enviar: {message}")
        
        # 1. Colocar em modo standby
        self.write_register(0x01, 0x81)
        time.sleep(0.1)
        
        # 2. Resetar ponteiro FIFO
        self.write_register(0x0D, 0x00)
        
        # 3. Escrever dados no FIFO
        print(f"Escrevendo {len(message)} bytes no FIFO...")
        for byte in message:
            self.write_register(0x00, byte)
        
        # 4. Configurar comprimento do payload
        self.write_register(0x22, len(message))
        
        # 5. Iniciar transmissão
        print("Iniciando transmissão...")
        self.write_register(0x01, 0x83)  # TX mode
        
        # 6. Aguardar transmissão com timeout
        timeout = 10  # segundos
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            irq_flags = self.read_register(0x12)
            
            if irq_flags & 0x08:  # TX_DONE
                print("✅ Transmissão concluída!")
                break
            elif irq_flags & 0x80:  # RX_TIMEOUT
                print("⚠️ Timeout na recepção detectado")
                break
                
            time.sleep(0.1)
        else:
            print("❌ Timeout na transmissão!")
            # Forçar volta ao standby
            self.write_register(0x01, 0x81)
            return False
        
        # 7. Limpar flag TX_DONE
        self.write_register(0x12, 0x08)
        
        # 8. Voltar para standby
        self.write_register(0x01, 0x81)
        
        print(f"✅ Mensagem enviada: {message}")
        return True
    
    def receive_message(self, timeout=10):
        """Receber mensagem via LoRa"""
        print("Iniciando recepção...")
        
        # Configurar modo recepção contínua
        self.write_register(0x01, 0x85)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            irq_flags = self.read_register(0x12)
            
            if irq_flags & 0x40:  # RX_DONE
                # Ler comprimento do pacote
                length = self.read_register(0x13)
                
                # Configurar ponteiro FIFO
                current_addr = self.read_register(0x10)
                self.write_register(0x0D, current_addr)
                
                # Ler dados
                data = []
                for i in range(length):
                    data.append(self.read_register(0x00))
                
                # Limpar flag
                self.write_register(0x12, 0x40)
                
                # Voltar para standby
                self.write_register(0x01, 0x81)
                
                received = bytes(data)
                print(f"✅ Mensagem recebida: {received}")
                return received
            
            time.sleep(0.1)
        
        # Timeout - voltar para standby
        self.write_register(0x01, 0x81)
        print("⏰ Timeout - nenhuma mensagem recebida")
        return None
    
    def close(self):
        """Fechar conexões"""
        self.spi.close()
        GPIO.cleanup()

