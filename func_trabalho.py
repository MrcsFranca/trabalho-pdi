import numpy as np
import cv2

from ativ1 import cinza
from ativ2 import suavizacao

# faz a suavizacao, depois o grandes por sobel, depois a supressao nao maxima e a limiarizacao dupla com histerese
def canny(img, limiar_baixo=50, limiar_alto=100, tam_abertura=3):
    img = cinza(img)
    return cv2.Canny(img, limiar_baixo, limiar_alto, apertureSize=tam_abertura)

# rotula componentes conexos em uma imagem binária e retorna os rotulos e o num de componentes. Mesma lógica da busca em largura de regiao()
def rotular_componentes(img_bin):
    h, w = img_bin.shape
    rotulos = np.zeros((h, w), dtype=np.int32)
    vizinhos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    rotulo_atual = 0

    for y in range(h):
        for x in range(w):
            if img_bin[y, x] == 255 and rotulos[y, x] == 0:
                rotulo_atual += 1
                fila = [(y, x)]
                rotulos[y, x] = rotulo_atual

                while len(fila) > 0:
                    y_atual, x_atual = fila.pop(0)

                    for dy, dx in vizinhos:
                        y_viz = y_atual + dy
                        x_viz = x_atual + dx
                        if 0 <= y_viz < h and 0 <= x_viz < w:
                            if img_bin[y_viz, x_viz] == 255 and rotulos[y_viz, x_viz] == 0:
                                rotulos[y_viz, x_viz] = rotulo_atual
                                fila.append((y_viz, x_viz))

    return rotulos, rotulo_atual
# Para cada componente encontrado calcula a área, perímetro, diâmetro e retorna uma lista de dicionários por obj encontrado
def medir_objetos(img_bin):
    rotulos, n = rotular_componentes(img_bin)
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    resultados = []

    for i in range(1, n + 1):
        mascara = np.where(rotulos == i, 255, 0).astype(np.uint8)

        area = int(np.sum(mascara == 255))

        erodida = cv2.erode(mascara, kernel_3x3)
        fronteira = cv2.subtract(mascara, erodida)
        ys, xs = np.where(fronteira == 255)
        perimetro = len(ys)

        diametro = 0.0
        if perimetro > 1:
            pontos = np.column_stack((xs, ys)).astype(np.float64)
            if len(pontos) > 1500:
                # fronteiras muito grandes: subamostra pra manter o cálculo rápido
                idx = np.linspace(0, len(pontos) - 1, 1500).astype(int)
                pontos = pontos[idx]
            diffs = pontos[:, None, :] - pontos[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
            diametro = float(np.max(dists))

        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        resultados.append({
            "id": i,
            "area": area,
            "perimetro": perimetro,
            "diametro": round(diametro, 2),
            "centro": (round(cx, 1), round(cy, 1)),
        })

    return resultados
