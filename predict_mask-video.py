# -*- coding: utf-8 -*-

# COVID-19 mask detector - https://github.com/alexcamargoweb/mask-detector
# Detecção de máscaras em pessoas (com ou sem)
# Adrian Rosebrock, COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning. PyImageSearch.    
# Disponível em: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/.   
# Acessado em: 05/12/2020.  
# Arquivo: predict_mask-video.py
# Execução via Spyder/Linux (pois o Google Colab não trabalha bem com acesso à câmera com OpenCV, somente em JS no momento)

# face detector
DETECTOR = './detector'
# final model
MODEL = './models/mask_covid-19.model'
# model confidence
CONFIDENCE = 0.5

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# cria uma função para detectar os frames do vídeo
def detect_and_predict_mask(frame, faceNet, maskNet):
	# pega as dimensões
	(h, w) = frame.shape[:2]
	# constrói um blob (objeto do tipo arquivo) da imagem
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	# passa o blob para a rede e obtém a detecção da face
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	# suas correspondentes localizações
	locs = []
	# e uma lista de detecções do doetector de faces
	preds = []

	# faz um loop sobre as detecções
	for i in range(0, detections.shape[2]):
		# extrai a confiança (por exemplo, a probabilidade) assossiada à detecção
		confidence = detections[0, 0, i, 2]
		# filtra as detecções fracas, garantindo que a confiança é
		# maior do que a confiança mínima
		if confidence > CONFIDENCE:
			# processa as coordenadas x e y da bouding box do objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# garante que as bouding boxes estejam dentro das dimensões do frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
	 
			# extrai o ROI do rosto, converte- do canal BGR para RGB
			# redimensiono para 224x224 e realiza o pré-processamento
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			
			# passa o rosto pelo modelo para determinar se
			# tem uma máscara ou não
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	 
	# só faz predições se ao menos um rosto foi detectado
	if len(faces) > 0:
		# para uma inferência mais rápida, faz previsões em lote em todas
		# faces ao mesmo tempo, em vez de previsões uma a uma no loop `for` acima
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# retorna uma tupla de 2 dos locais dos rostos
	return (locs, preds)

# INICIA A DETECÇÃO

# carrega o detector de faces
print("[INFO] carregando detector de faces...")
# estrutura da rede
prototxtPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
# pesos da rede
weightsPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# carrega do disco o modelo detector de máscaras
print("[INFO] carregando detector de faces...")
maskNet = load_model(MODEL)
# inicializa o stream de vídeo e permite que o sensor da câmera funcione
print("[INFO] iniciando transmissão de vídeo...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

# faz um loop sobre as detecções do vídeo
while True:
	# pega o frame do stream de vídeo redimensiona-o para uma largura máxima de 800 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width = 800)
	# detecta as faces do frame e determina se estão usando máscaras, ou não
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
 
	# faz um loop sobre as faces detectadas e suas respectivas localizações
	for (box, pred) in zip(locs, preds):
		# armazena a bounding box e as predições
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determina o label da classe e a cor para desenhar
		# a bouding box e o texto
		label = "Com mascara" if mask > withoutMask else "Sem mascara"
		color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)
		# inclui a probabilidade no label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# exibe o label e o retângulo da bounding box no frame de saída
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	
	# exibe o frame de saída
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# se a tecla "q" for pressionada, sai do loop de execução
	if key == ord("q"):
		break
# limpa a execução
cv2.destroyAllWindows()
vs.stop()
