import cv2
import numpy as np
import matplotlib.pyplot as plt

def cinza(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return img

def mediana(img, tam):
    if isinstance(img, str):
        img = cinza(img)

    h,w = img.shape
    suave = np.zeros_like(img, dtype=np.uint8)
    margem = tam // 2
    img_pad = np.pad(img, margem, mode='reflect')

    for y in range(h):
        for x in range(w):
            vizin = img_pad[y:y+tam, x:x+tam].flatten()
            suave[y, x] = np.median(vizin)

    return suave

    img = cinza(img)

def pontos_isolados(img, limiar):
    img = cinza(img)

    mascara = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]], dtype=np.float32)
    R = cv2.filter2D(img.astype(np.float32), ddepth=cv2.CV_32F, kernel=mascara)
    R_abs = np.abs(R)
    img_resultado = np.where(R_abs > limiar, 255, 0).astype(np.uint8)

    return img_resultado

def detectar_linhas(img, limiar):
    img = cinza(img)

    m_horizontal = np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]], dtype=np.float32)
    m_mais_45 = np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]], dtype=np.float32)
    m_vertical = np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]], dtype=np.float32)
    m_menos_45 = np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]], dtype=np.float32)

    mascaras = {
        "Horizontal": m_horizontal,
        "+45°": m_mais_45,
        "Vertical": m_vertical,
        "-45°": m_menos_45
    }

    resultados_limiarizados = {}

    for nome, mascara in mascaras.items():
        R = cv2.filter2D(img.astype(np.float32), ddepth=cv2.CV_32F, kernel=mascara)
        R_abs = np.abs(R)

        img_limiar = np.where(R_abs > limiar, 255, 0).astype(np.uint8)
        resultados_limiarizados[nome] = img_limiar

        cv2.imwrite(f"linhas_{nome.replace('°', '')}.png", resultados_limiarizados[nome])

    res_final = cv2.bitwise_or(resultados_limiarizados["Horizontal"], resultados_limiarizados["+45°"])
    res_final = cv2.bitwise_or(res_final, resultados_limiarizados["Vertical"])
    res_final = cv2.bitwise_or(res_final, resultados_limiarizados["-45°"])

    return res_final, resultados_limiarizados

def regiao(img, seed, threshold):

    h, w = img.shape
    mascara = np.zeros((h, w), dtype=np.uint8)

    y_sem, x_sem= seed

    intensidade_semente = int(img[y_sem, x_sem])
    fila = [(y_sem, x_sem)]
    mascara[y_sem, x_sem] = 255

    vizinhos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]

    while len(fila) > 0:
        y_atual, x_atual = fila.pop(0)

        for dy, dx in vizinhos:
            y_vizinho = y_atual + dy
            x_vizinho = x_atual+dx
            if 0 <= y_vizinho < h and 0 <= x_vizinho < w:
                if mascara[y_vizinho, x_vizinho] == 0:
                    intensidade_vizinho =int(img[y_vizinho, x_vizinho])

                    if abs(intensidade_vizinho - intensidade_semente) < threshold:
                        mascara[y_vizinho, x_vizinho] = 255
                        fila.append((y_vizinho, x_vizinho))

    return mascara

def otsu(img):
    img = cinza(img)

    hist, _ = np.histogram(img, bins=256, range=(0, 256))

    total_pixels = img.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_b = 0
    w_b = 0
    max_variance = 0
    limiar = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue

        w_f = total_pixels - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]

        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_entre = w_b * w_f * (m_b - m_f) ** 2

        if var_entre > max_variance:
            max_variance = var_entre
            limiar = t

    img_bin = np.where(img >= limiar, 255, 0).astype(np.uint8)

    return limiar, img_bin

