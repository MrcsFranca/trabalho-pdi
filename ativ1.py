import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cinza(img):
    # Para aplicar o grayscale
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return img

def negativa(img):
    # Para implementar imagem negativa
    img = cv2.imread(img)
    img = 255 - img
    return img

def contraste(img, c, d):
    img = cv2.imread(img)

    img = img.astype(np.float32)

    a = np.min(img)
    b = np.max(img)

    img_contraste = (img - a) * ((d - c) / (b - a)) + c
    img_contraste = np.clip(img_contraste, 0, 255)   
    img_contraste = np.astype(img_contraste, 'uint8')
    return img_contraste

# a func logaritmo comprime faixa de intensidade alta e expande de intensidade baixa -> realça detalhes escuros
def logaritmo(img):
    img = cv2.imread(img)

    img = img.astype(np.float32)

    R = np.max(img)
    c = 255 / (np.log(1 + R))

    img = c * np.log(1 + np.abs(img))

    img = np.astype(img, 'uint8')
    
    return img

def potencia(img, c, gama):
    img = cv2.imread(img)
    img = img.astype(np.float32)

    img = img / 255.0
    img = c * (img ** gama)
    img = img * 255.0

    img = np.clip(img, 0, 255) # usei o clip pq o valor pode ser maior que 1 e estourar o 255
    img = img.astype(np.uint8)
    return img

# vai fazendo o shift left para pegar as camadas dos bits
def fatiamento(img, saida):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    os.makedirs('results/fatiamento', exist_ok=True)
    planos = []

    for i in range(8):
        mascara = 1 << i
        plano = img & mascara
        visivel = (plano >> i) * 255

        nome_arquivo = f'results/fatiamento/{saida}_bit_{i}.png'
        cv2.imwrite(nome_arquivo, visivel)
        planos.append(visivel)

    return planos

def calc_hist(img):
    hist = np.zeros(256, dtype=np.int32)

    pixels = img.ravel()

    for pixel in pixels:
        hist[pixel] += 1

    return hist

def calc_hist_normalizado(img):
    hist = calc_hist(img)

    total = img.size

    return hist / total

def calc_hist_acumulado(img):
    hist = calc_hist(img)
    hist_acumulado = np.zeros(256, dtype=np.int32)

    soma = 0
    for i in range(256):
        soma += hist[i]
        hist_acumulado[i] = soma
    return hist_acumulado

def hist_normalizado_acumulado(img):
    hist_acumulado = calc_hist_acumulado(img)
    total = img.size

    return hist_acumulado / total

def equalizar_hist(img):
    ha = calc_hist_acumulado(img)
    MN = img.size

    valores_existentes = ha[ha > 0]
    min_ha = np.min(valores_existentes)

    if MN == min_ha:
        lut = np.zeros(256, dtype='uint8')
    else:
        lut = ((ha - min_ha) / (MN - min_ha)) * 255
        
        lut = np.clip(lut, 0, 255)
        lut = np.round(lut).astype('uint8')
    
    img_equalizada = lut[img]

    return img_equalizada

#gerado por ia apenas para estilizar o histograma
def configurar_eixo_modelo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=14, pad=10, fontname='sans-serif')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-5, 260)
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

