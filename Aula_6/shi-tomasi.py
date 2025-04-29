import cv2
import numpy as np

# Carregar a imagem em escala de cinza
imagem = cv2.imread('exemplo_cozinha.jpg', cv2.IMREAD_GRAYSCALE)

if imagem is None:
    print("Erro: Não foi possível carregar a imagem.")
    exit()

# Função para aplicar Shi-Tomasi com diferentes parâmetros
def aplicar_shi_tomasi(imagem, max_corners, quality_level, min_distance):
    # Aplicar o detector Shi-Tomasi
    cantos = cv2.goodFeaturesToTrack(imagem, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    cantos = np.int32(cantos)
    
    # Criar uma cópia da imagem para desenhar os cantos
    imagem_cantos = imagem.copy()
    for canto in cantos:
        x, y = canto.ravel()
        cv2.circle(imagem_cantos, (x, y), 3, 255, -1)
    return imagem_cantos

# Aplicar Shi-Tomasi com diferentes configurações
imagem_shi_tomasi_1 = aplicar_shi_tomasi(imagem, max_corners=50, quality_level=0.01, min_distance=10)
imagem_shi_tomasi_2 = aplicar_shi_tomasi(imagem, max_corners=100, quality_level=0.05, min_distance=20)
imagem_shi_tomasi_3 = aplicar_shi_tomasi(imagem, max_corners=150, quality_level=0.1, min_distance=30)

# Exibir resultados em janelas separadas
cv2.imshow('Shi-Tomasi - Configuração 1', imagem_shi_tomasi_1)
cv2.imshow('Shi-Tomasi - Configuração 2', imagem_shi_tomasi_2)
cv2.imshow('Shi-Tomasi - Configuração 3', imagem_shi_tomasi_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
