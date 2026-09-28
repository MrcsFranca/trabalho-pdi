#wrapper para as funções. so ajusta o formato de entrada e saida
import io
import cv2
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")  # backend sem janela para n travar a UI
import matplotlib.pyplot as plt

from ativ1 import (
    cinza, negativa, calc_hist, equalizar_hist,
    contraste, logaritmo, potencia, fatiamento,
    calc_hist_normalizado, calc_hist_acumulado, hist_normalizado_acumulado,
)
from ativ2 import suavizacao, kvizinhos, sobel, prewitt, roberts, laplaciano
from ativ3 import espectro_fourier, filtragem_frequencia, rejeita, filtragem
from ativ4 import mediana as mediana_ativ4, otsu, pontos_isolados, detectar_linhas, regiao
from ativ5 import abertura, fechamento, fronteira_interna, fronteira_externa, gradiente, preenchimento, componentes
from func_trabalho import canny, rotular_componentes, medir_objetos

# em algumas funções eu poderia passar um valor fora de 0 e 255, então fiz isso como tratativa de erro
def normalizar_0_255(img_float):
    a = np.min(img_float)
    b = np.max(img_float)
    if b == a:
        return np.zeros_like(img_float, dtype=np.uint8)
    return np.clip((img_float - a) / (b - a) * 255, 0, 255).astype(np.uint8)

#vonverte uma imagem no formato do openCV para um formato que o textual aceita
def cv2_para_pil(img_cv2):
    if len(img_cv2.shape) == 2:
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(img_rgb)

# func para montar o elementro estruturante
def kernel(tam):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (tam, tam))

# função para "desenha" o histograma
def plotar_histograma(hist, titulo="Histograma", ylabel="Pixels"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(256), hist, width=0.8, color="#4F81BD")
    ax.set_title(titulo)
    ax.set_xlabel("Nível de cinza")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert("RGB")

# parecido com o histograma mas para mostrar as imagens de fatiamento
def plotar_fatiamento(planos):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(planos[i], cmap='gray')
        ax.set_title(f"Bit {i}")
        ax.axis('off')
    fig.suptitle("Fatiamento de planos de bits")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert("RGB")

# criei uma imagem colorida para ficar mais fácil de observar os objetos encontrados pelo componentes conexos
def colorir_rotulos(rotulos, n):
    rng = np.random.default_rng(42)  # isso aqui é para ter as mesmas cores toda vez que rodar
    cores = rng.integers(50, 255, size=(n + 1, 3))
    cores[0] = 0  # rótulo 0 = fundo
    return cores[rotulos].astype(np.uint8)

# imagem colorida dos objetos com a tabela de área, perímetro e diâmetro, é a mesma ideia de plotar o histograma, só tive que adaptar
def plotar_medicoes(img_bin, titulo="Objetos detectados", max_linhas=20):
    rotulos, n = rotular_componentes(img_bin)
    resultados = medir_objetos(img_bin)
    colorida = colorir_rotulos(rotulos, n)

    resultados_ordenados = sorted(resultados, key=lambda r: r["area"], reverse=True)
    mostrados = resultados_ordenados[:max_linhas]

    fig, (ax_img, ax_tab) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_img.imshow(colorida)
    ax_img.set_title(f"{titulo} ({n} objeto(s) no total)")
    ax_img.axis("off")

    ax_tab.axis("off")
    if mostrados:
        linhas = [[r["id"], r["area"], r["perimetro"], r["diametro"]] for r in mostrados]
        tabela = ax_tab.table(
            cellText=linhas,
            colLabels=["ID", "Área (px)", "Perímetro (px)", "Diâmetro (px)"],
            loc="center",
        )
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(9)
        tabela.scale(1, 1.3)
        if n > max_linhas:
            ax_tab.set_title(f"{max_linhas} maiores de {n} (ordenados por área)", fontsize=9)
    else:
        ax_tab.text(0.5, 0.5, "Nenhum objeto encontrado", ha="center", va="center")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert("RGB")

def parse_int(valor, default):
    try:
        return int(valor.strip())
    except (ValueError, AttributeError):
        return default

def parse_float_pair(valor, default):
    try:
        a, b = valor.split(",")
        return float(a), float(b)
    except Exception:
        return default

def parse_pair(valor, default):
    try:
        a, b = valor.split(",")
        return int(a), int(b)
    except Exception:
        return default

def parse_triple(valor, default):
    try:
        a, b, c = valor.split(",")
        return int(a), int(b), int(c)
    except Exception:
        return default

# handlers para as operações em si
def op_original(caminho, p):
    return cv2.imread(caminho)

def op_cinza(caminho, p):
    return cinza(caminho)

def op_negativa(caminho, p):
    return negativa(caminho)

def op_otsu(caminho, p):
    _limiar, img_bin = otsu(caminho)
    return img_bin

def op_media(caminho, p):
    tam = parse_int(p, 3)
    _, resultado = suavizacao(caminho, tam)
    return resultado

def op_mediana(caminho, p):
    tam = parse_int(p, 3)
    return mediana_ativ4(caminho, tam)

def op_kvizinhos(caminho, p):
    tam, k = parse_pair(p, (5, 3))
    _, resultado = kvizinhos(caminho, tam, k)
    return resultado

def op_sobel(caminho, p):
    _, resultado = sobel(caminho)
    return resultado

def op_prewitt(caminho, p):
    _, resultado = prewitt(caminho)
    return resultado

def op_roberts(caminho, p):
    _, resultado = roberts(caminho)
    return resultado

def op_laplaciano(caminho, p):
    _, resultado = laplaciano(caminho)
    return resultado

def op_erosao(caminho, p):
    tam = parse_int(p, 3)
    return cv2.erode(cinza(caminho), kernel(tam))

def op_dilatacao(caminho, p):
    tam = parse_int(p, 3)
    return cv2.dilate(cinza(caminho), kernel(tam))

def op_abertura(caminho, p):
    tam = parse_int(p, 3)
    return abertura(cinza(caminho), kernel(tam))

def op_fechamento(caminho, p):
    tam = parse_int(p, 3)
    return fechamento(cinza(caminho), kernel(tam))

def op_gradiente(caminho, p):
    tam = parse_int(p, 3)
    return gradiente(cinza(caminho), kernel(tam))

def op_fronteira_int(caminho, p):
    tam = parse_int(p, 3)
    return fronteira_interna(cinza(caminho), kernel(tam))

def op_fronteira_ext(caminho, p):
    tam = parse_int(p, 3)
    return fronteira_externa(cinza(caminho), kernel(tam))

def op_histograma(caminho, p):
    return plotar_histograma(calc_hist(cinza(caminho)))

def op_equalizar(caminho, p):
    return equalizar_hist(cinza(caminho))

def op_pontos_isolados(caminho, p):
    limiar = parse_int(p, 200)
    return pontos_isolados(caminho, limiar)

def op_linhas(caminho, p):
    limiar = parse_int(p, 200)
    resultado, _parcial = detectar_linhas(caminho, limiar)
    return resultado

def op_regiao(caminho, p):
    x, y, limiar = parse_triple(p, (10, 10, 15))
    # regiao espera seed como (y, x)
    return regiao(cinza(caminho), (y, x), limiar)

def op_preenchimento(caminho, p):
    x, y = parse_pair(p, (0, 0))
    img = cinza(caminho)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # preenchimento espera seed como (x, y)
    return preenchimento(img_bin, (x, y))

def op_componentes_semente(caminho, p):
    x, y = parse_pair(p, (0, 0))
    img = cinza(caminho)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # componentes também espera seed como (x, y)
    mascara_componente = componentes(img_bin, (x, y))

    resultado_amarelo = cv2.cvtColor(mascara_componente, cv2.COLOR_GRAY2BGR)
    resultado_amarelo[mascara_componente == 255] = [0, 255, 255]  # para colocar o item em amarelo
    return resultado_amarelo

def op_canny(caminho, p):
    limiar_baixo, limiar_alto, tam_abertura = parse_triple(p, (100, 200, 3))
    return canny(caminho, limiar_baixo, limiar_alto, tam_abertura)

def op_componentes_conexos(caminho, p):
    _limiar, img_bin = otsu(caminho)
    rotulos, n = rotular_componentes(img_bin)
    colorida = colorir_rotulos(rotulos, n)
    cv2.putText(
        colorida, f"{n} objeto(s) encontrado(s)", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return colorida

def op_medir_objetos(caminho, p):
    _limiar, img_bin = otsu(caminho)
    return plotar_medicoes(img_bin)

def op_contraste(caminho, p):
    c, d = parse_pair(p, (0, 255))
    return contraste(caminho, c, d)

def op_logaritmo(caminho, p):
    return logaritmo(caminho)

def op_potencia(caminho, p):
    c, gama = parse_float_pair(p, (1.0, 1.0))
    return potencia(caminho, c, gama)

def op_fatiamento(caminho, p):
    nome_saida = p.strip() if p.strip() else "img"
    planos = fatiamento(caminho, nome_saida)
    return plotar_fatiamento(planos)

def op_hist_normalizado(caminho, p):
    return plotar_histograma(
        calc_hist_normalizado(cinza(caminho)),
        titulo="Histograma normalizado",
        ylabel="Frequência relativa",
    )

def op_hist_acumulado(caminho, p):
    return plotar_histograma(
        calc_hist_acumulado(cinza(caminho)),
        titulo="Histograma acumulado",
        ylabel="Pixels acumulados",
    )

def op_hist_acumulado_normalizado(caminho, p):
    return plotar_histograma(
        hist_normalizado_acumulado(cinza(caminho)),
        titulo="Histograma acumulado normalizado",
        ylabel="Frequência acumulada",
    )

def op_espectro_fourier(caminho, p):
    espectro = espectro_fourier(caminho)
    return normalizar_0_255(espectro)

def op_gaussiano_baixa(caminho, p):
    d0 = parse_int(p, 30)
    img_cinza = cinza(caminho)
    img_low, _img_high = filtragem_frequencia(img_cinza, d0)
    return normalizar_0_255(img_low)

def op_gaussiano_alta(caminho, p):
    d0 = parse_int(p, 30)
    img_cinza = cinza(caminho)
    _img_low, img_high = filtragem_frequencia(img_cinza, d0)
    return normalizar_0_255(img_high)

def op_rejeita_mascara(caminho, p):
    caminho_mascara = p.strip()
    if not caminho_mascara:
        raise ValueError("Imagem de máscara vai no campo parâmetro")
    forma_img = cinza(caminho).shape
    forma_mascara = cinza(caminho_mascara).shape
    if forma_img != forma_mascara:
        raise ValueError(
            f"Máscara e imagem precisam ter o mesmo tamanho "
                f"(imagem: {forma_img[1]}x{forma_img[0]}, máscara: {forma_mascara[1]}x{forma_mascara[0]})"
        )

    _img, img_back, _spectrum, _m = rejeita(caminho, caminho_mascara)
    return normalizar_0_255(img_back)

def op_banda_passa(caminho, p):
    d0, W = parse_pair(p, (60, 30))
    _img, _espectro, _Hp, _Hr, img_passa, _img_rejeita = filtragem(caminho, d0, W)
    return normalizar_0_255(img_passa)

def op_banda_rejeita(caminho, p):
    d0, W = parse_pair(p, (60, 30))
    _img, _espectro, _Hp, _Hr, _img_passa, img_rejeita = filtragem(caminho, d0, W)
    return normalizar_0_255(img_rejeita)

#dicionario para chamar as funções mais facilmente na main
OPERACOES = {
    "op_original": op_original,
    "op_cinza": op_cinza,
    "op_negativa": op_negativa,
    "op_otsu": op_otsu,
    "op_media": op_media,
    "op_mediana": op_mediana,
    "op_kvizinhos": op_kvizinhos,
    "op_sobel": op_sobel,
    "op_prewitt": op_prewitt,
    "op_roberts": op_roberts,
    "op_laplaciano": op_laplaciano,
    "op_erosao": op_erosao,
    "op_dilatacao": op_dilatacao,
    "op_abertura": op_abertura,
    "op_fechamento": op_fechamento,
    "op_gradiente": op_gradiente,
    "op_fronteira_int": op_fronteira_int,
    "op_fronteira_ext": op_fronteira_ext,
    "op_histograma": op_histograma,
    "op_equalizar": op_equalizar,
    "op_pontos_isolados": op_pontos_isolados,
    "op_linhas": op_linhas,
    "op_regiao": op_regiao,
    "op_preenchimento": op_preenchimento,
    "op_componentes_semente": op_componentes_semente,
    "op_canny": op_canny,
    "op_componentes_conexos": op_componentes_conexos,
    "op_medir_objetos": op_medir_objetos,
    "op_contraste": op_contraste,
    "op_logaritmo": op_logaritmo,
    "op_potencia": op_potencia,
    "op_fatiamento": op_fatiamento,
    "op_hist_normalizado": op_hist_normalizado,
    "op_hist_acumulado": op_hist_acumulado,
    "op_hist_acumulado_normalizado": op_hist_acumulado_normalizado,
    "op_espectro_fourier": op_espectro_fourier,
    "op_gaussiano_baixa": op_gaussiano_baixa,
    "op_gaussiano_alta": op_gaussiano_alta,
    "op_rejeita_mascara": op_rejeita_mascara,
    "op_banda_passa": op_banda_passa,
    "op_banda_rejeita": op_banda_rejeita,
}
