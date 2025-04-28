import cv2
import numpy as np

# Carregar a imagem em escala de cinza
imagem = cv2.imread('exemplo_cozinha.jpg', cv2.IMREAD_GRAYSCALE)

if imagem is None:
    print("Erro: Não foi possível carregar a imagem.")
    exit()

# Aplicar o detector de Harris com diferentes parâmetros
harris1 = cv2.cornerHarris(imagem, blockSize=2, ksize=3, k=0.04)
harris1_dilatado = cv2.dilate(harris1, None)

harris2 = cv2.cornerHarris(imagem, blockSize=4, ksize=5, k=0.06)
harris2_dilatado = cv2.dilate(harris2, None)

harris3 = cv2.cornerHarris(imagem, blockSize=6, ksize=7, k=0.08)
harris3_dilatado = cv2.dilate(harris3, None)

# Criar cópias da imagem para exibir os resultados
imagem_harris1 = imagem.copy()
imagem_harris2 = imagem.copy()
imagem_harris3 = imagem.copy()

imagem_harris1[harris1_dilatado > 0.01 * harris1_dilatado.max()] = [255]
imagem_harris2[harris2_dilatado > 0.01 * harris2_dilatado.max()] = [255]
imagem_harris3[harris3_dilatado > 0.01 * harris3_dilatado.max()] = [255]

# Exibir resultados em janelas separadas
cv2.imshow('Harris Corner - Configuração 1', imagem_harris1)
cv2.imshow('Harris Corner - Configuração 2', imagem_harris2)
cv2.imshow('Harris Corner - Configuração 3', imagem_harris3)
cv2.waitKey(0)
cv2.destroyAllWindows()
