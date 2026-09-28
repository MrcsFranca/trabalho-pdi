import cv2
import numpy as np
import matplotlib.pyplot as plt

def cinza(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return img

def espectro_fourier(img):
    img = cinza(img)

    f = np.fft.fft2(img)

    fshift = np.fft.fftshift(f)

    mag = np.abs(fshift)

    espectro_log = 20 * np.log(1 + mag)

    return espectro_log

def filtros_gaussiano(shape, d0):
    M, N = shape
    u = np.arange(M)
    v = np.arange(N)

    u = u - M / 2
    v = v - N / 2

    U, V = np.meshgrid(v, u)

    D = np.sqrt(U**2 + V**2)

    H_low = np.exp(-(D**2) / (2 * (d0**2)))

    H_high = 1 - H_low

    return H_low, H_high

def filtragem_frequencia(img, d0=30):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    H_low, H_high = filtros_gaussiano(img.shape, d0)

    fshift_low = fshift * H_low
    fshift_high = fshift * H_high

    f_ishift_low = np.fft.ifftshift(fshift_low)
    f_ishift_high = np.fft.ifftshift(fshift_high)

    img_back_low = np.fft.ifft2(f_ishift_low)
    img_back_high = np.fft.ifft2(f_ishift_high)

    img_low = np.abs(img_back_low)
    img_high = np.abs(img_back_high)

    return img_low, img_high

def rejeita(img, m):
    img = cinza(img)
    m = cinza(m)

    m = m > 0
    m = m.astype('uint8')

    rows, cols = m.shape
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[:, :, 0] = m
    mask[:, :, 1] = m

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    spectrum = np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    filtered_shift = dft_shift * mask

    filtered = np.fft.ifftshift(filtered_shift)
    img_back = cv2.idft(filtered)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img, img_back, spectrum, m

def filtro_banda(shape, d0, W):
    M, N = shape
    u = np.arange(M) - M / 2
    v = np.arange(N) - N / 2
    U, V = np.meshgrid(v, u)

    D = np.sqrt(U**2 + V**2)

    dentro = (D >= (d0 - W / 2)) & (D <= (d0 + W / 2))

    H_passa = np.zeros((M, N))
    H_passa[dentro] = 1

    H_rejeita = 1 - H_passa

    return H_passa, H_rejeita

def filtragem(img, d0=60, W=30):
    img = cinza(img)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    espectro = 20 * np.log(1 + np.abs(fshift))

    H_passa, H_rejeita = filtro_banda(img.shape, d0, W)

    fshift_passa = fshift * H_passa
    fshift_rejeita = fshift * H_rejeita

    img_passa = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_passa)))
    img_rejeita = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_rejeita)))

    return img, espectro, H_passa, H_rejeita, img_passa, img_rejeita

