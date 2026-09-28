import cv2 as cv
import numpy as np

def abertura(img, kernel):
    img_erodida = cv.erode(img, kernel)
    img_restaurada = cv.dilate(img_erodida, kernel)
    return img_restaurada

def fechamento(img, kernel):
    img_dilatada = cv.dilate(img, kernel)
    img_restaurada = cv.erode(img_dilatada, kernel)
    return img_restaurada

def fronteira_interna(img, kernel):
    img_erodida = cv.erode(img, kernel)
    return cv.subtract(img, img_erodida)

def fronteira_externa(img, kernel):
    img_diltatada = cv.dilate(img, kernel)
    return cv.subtract(img_diltatada, img)

def preenchimento(img, seed):
    complemento = cv.bitwise_not(img)

    x_atual = np.zeros_like(img)
    x_atual[seed[0], seed[1]] = 255

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        img_dilatada = cv.dilate(x_atual, kernel)
        x_proximo = cv.bitwise_and(img_dilatada, complemento)

        if np.array_equal(x_atual, x_proximo):
            break

        x_atual = x_proximo

    result = cv.bitwise_or(img, x_atual)
    return result

def componentes(img, seed):
    x_atual = np.zeros_like(img)

    x_x, y_y = seed
    x_atual[y_y, x_x] = 255

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    while True:
        img_dilatada = cv.dilate(x_atual, kernel)
        x_proximo = cv.bitwise_and(img_dilatada, img)

        if np.array_equal(x_atual, x_proximo):
            break

        x_atual = x_proximo

    return x_atual

def gradiente(img, kernel):
    img_dilatada = cv.dilate(img, kernel)
    img_erodida = cv.erode(img, kernel)
    return cv.subtract(img_dilatada, img_erodida)

