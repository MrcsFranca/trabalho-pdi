import cv2
import numpy as np
import matplotlib.pyplot as plt

def cinza(img):
    # Para aplicar o grayscale
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return img

def suavizacao(img, tam):
    img = cinza(img)

    M = tam * tam
    margem = tam // 2

    img_pad = np.pad(img, margem, mode='reflect')
    img_suave = np.zeros_like(img, dtype=np.uint8)

    # calcula a média pela vizinhança
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            vizin = img_pad[y:y + tam, x:x + tam]
            img_suave[y, x] = np.mean(vizin)

    return img, img_suave

# essa função foi desenvolvida com o auxílio de IA. Não tinha entendido muito bem como ela funcionava
def kvizinhos(img, tam, k):
    img = cinza(img)

    h, w = img.shape

    suave = np.zeros_like(img, dtype=np.uint8)
    margem = tam // 2
    img_pad = np.pad(img, margem, mode='reflect')

    for y in range(h):
        for x in range(w):
            central = img[y, x]
            vizin = img_pad[y:y+tam, x:x+tam].flatten()

            diferenca = np.abs(vizin.astype(np.int32) - int(central))

            idx_proximos = np.argsort(diferenca)[:k]
            val_proximos = vizin[idx_proximos]

            suave[y, x] = np.mean(val_proximos)

    return img, suave

def mediana(img, tam):
    img = cinza(img)

    h,w = img.shape
    suave = np.zeros_like(img, dtype=np.uint8)
    margem = tam // 2
    img_pad = np.pad(img, margem, mode='reflect')

    for y in range(h):
        for x in range(w):
            vizin = img_pad[y:y+tam, x:x+tam].flatten()
            suave[y, x] = np.median(vizin)

    return img, suave

def laplaciano(img):
    img = cinza(img)

    kernel = np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]], dtype=np.float32)

    img_laplace = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_DEFAULT)

    img_laplace = np.absolute(img_laplace)
    img_laplace = np.clip(img_laplace, 0, 255).astype(np.uint8)

    return img, img_laplace

def roberts(img):
    img = cinza(img)

    h1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    h2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    g1 = cv2.filter2D(img, cv2.CV_32F, h1)
    g2 = cv2.filter2D(img, cv2.CV_32F, h2)

    Mag = np.sqrt(g1**2 + g2**2)

    Mag = np.clip(Mag, 0, 255).astype(np.uint8)

    return img, Mag

def prewitt(img):
    img = cinza(img)

    h1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    h2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

    g1 = cv2.filter2D(img, cv2.CV_32F, h1)
    g2 = cv2.filter2D(img, cv2.CV_32F, h2)

    Mag = np.sqrt(g1**2 + g2**2)
    
    Mag = np.clip(Mag, 0, 255).astype(np.uint8)
    
    return img, Mag

# o kernel foi implementado por mim e a convoluçaõ eu usei o filter2D 
def sobel(img):
    img = cinza(img)

    h1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    h2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    g1 = cv2.filter2D(img, cv2.CV_32F, h1)
    g2 = cv2.filter2D(img, cv2.CV_32F, h2)

    Mag = np.sqrt(g1**2 + g2**2)
    
    Mag = np.clip(Mag, 0, 255).astype(np.uint8)
    
    return img, Mag

