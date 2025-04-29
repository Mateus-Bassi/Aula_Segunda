import cv2

# Carregar a imagem em escala de cinza
imagem = cv2.imread('exemplo_banheiro.jpg', cv2.IMREAD_GRAYSCALE)

if imagem is None:
    print("Erro: Não foi possível carregar a imagem.")
    exit() 

# Criar o objeto SIFT
sift = cv2.SIFT_create()

# Detectar e computar características
keypoints, descriptors = sift.detectAndCompute(imagem, None)

# Desenhar os keypoints na imagem
imagem_sift = cv2.drawKeypoints(imagem, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Exibir resultados
cv2.imshow('SIFT - exemplo_cozinha', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()