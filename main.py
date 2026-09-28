import os
import cv2
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ListView, ListItem, Label, Input
from textual.containers import Horizontal, Vertical
from textual_image.widget import Image as ImagemTerminal
from PIL import Image as PILImage

from operacoes import OPERACOES, cv2_para_pil

# cache para somar os algoritmos
CACHE_DIR = "_cache"
CACHE_PATH = os.path.join(CACHE_DIR, "temp.png")

# passa o número digitado apara o id da operação para tecla de atalho
ATALHOS = [
    "op_original", "op_cinza", "op_negativa", "op_otsu",
    "op_media", "op_mediana", "op_kvizinhos",
    "op_sobel", "op_prewitt", "op_roberts", "op_laplaciano",
    "op_erosao", "op_dilatacao", "op_abertura", "op_fechamento",
    "op_gradiente", "op_fronteira_int", "op_fronteira_ext",
    "op_histograma", "op_equalizar",
    "op_pontos_isolados", "op_linhas", "op_regiao",
    "op_preenchimento", "op_componentes_semente",
    "op_canny", "op_componentes_conexos", "op_medir_objetos",
    "op_contraste", "op_logaritmo", "op_potencia", "op_fatiamento",
    "op_hist_normalizado", "op_hist_acumulado", "op_hist_acumulado_normalizado",
    "op_espectro_fourier", "op_gaussiano_baixa", "op_gaussiano_alta",
    "op_rejeita_mascara", "op_banda_passa", "op_banda_rejeita",
]

TEMPO_ESPERA_SEGUNDO_DIGITO = 0.6

def salvar_como_atual(resultado):
    # salva o resultado da ultima operacao em disco para usar como entrada para proxima
    os.makedirs(CACHE_DIR, exist_ok=True)
    if isinstance(resultado, PILImage.Image):
        resultado.save(CACHE_PATH)
    else:
        cv2.imwrite(CACHE_PATH, resultado)
    return CACHE_PATH


class TrabalhoPDI(App):
    CSS = """
    Screen { layout: horizontal; }
    #menu-lateral {
        width: 38%;
        dock: left;
        border-right: solid green;
        padding: 1;
    }
    #area-principal { width: 62%; align: center middle; }
    #tela_render { width: auto; height: auto; border: solid cyan; padding: 1; }
    Input { margin-bottom: 1; }
    .titulo { text-style: bold; color: yellow; }
    .legenda { color: grey; margin-top: 1; }
    #digitando { color: cyan; }
    """

    BINDINGS = [("q", "quit", "Sair")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="menu-lateral"):
                yield Label("Caminho da Imagem:", classes="titulo")
                yield Input(placeholder="Caminho da imagem", id="input_img")

                yield Label("Parâmetro:", classes="titulo")
                yield Input(placeholder="parâmentros para operação", id="input_param")

                yield Label("Operações", classes="titulo")
                yield ListView(
                    ListItem(Label("0. Original — reseta os filtros"), id="op_original"),
                    ListItem(Label("1. Cinza"), id="op_cinza"),
                    ListItem(Label("2. Negativo"), id="op_negativa"),
                    ListItem(Label("3. Otsu"), id="op_otsu"),
                    ListItem(Label("4. Suavização - Média [tam]"), id="op_media"),
                    ListItem(Label("5. Suavização - Mediana [tam]"), id="op_mediana"),
                    ListItem(Label("6. Suavização - k-vizinhos [tam,k]"), id="op_kvizinhos"),
                    ListItem(Label("7. Bordas - Sobel"), id="op_sobel"),
                    ListItem(Label("8. Bordas - Prewitt"), id="op_prewitt"),
                    ListItem(Label("9. Bordas - Roberts"), id="op_roberts"),
                    ListItem(Label("10. Bordas - Laplaciano"), id="op_laplaciano"),
                    ListItem(Label("11. Erosão [tam]"), id="op_erosao"),
                    ListItem(Label("12. Dilatação [tam]"), id="op_dilatacao"),
                    ListItem(Label("13. Abertura [tam]"), id="op_abertura"),
                    ListItem(Label("14. Fechamento [tam]"), id="op_fechamento"),
                    ListItem(Label("15. Gradiente morfológico [tam]"), id="op_gradiente"),
                    ListItem(Label("16. Fronteira interna [tam]"), id="op_fronteira_int"),
                    ListItem(Label("17. Fronteira externa [tam]"), id="op_fronteira_ext"),
                    ListItem(Label("18. Histograma"), id="op_histograma"),
                    ListItem(Label("19. Equalizar histograma"), id="op_equalizar"),
                    ListItem(Label("20. Pontos isolados [limiar]"), id="op_pontos_isolados"),
                    ListItem(Label("21. Detecção de linhas [limiar]"), id="op_linhas"),
                    ListItem(Label("22. Crescimento de região [x,y,limiar]"), id="op_regiao"),
                    ListItem(Label("23. Preenchimento de região [x,y]"), id="op_preenchimento"),
                    ListItem(Label("24. Componente c/ semente [x,y]"), id="op_componentes_semente"),
                    ListItem(Label("25. Canny [limiar_baixo,limiar_alto,tam]"), id="op_canny"),
                    ListItem(Label("26. Componentes conexos (contagem)"), id="op_componentes_conexos"),
                    ListItem(Label("27. Área/perímetro/diâmetro"), id="op_medir_objetos"),
                    ListItem(Label("28. Contraste (normalização) [c,d]"), id="op_contraste"),
                    ListItem(Label("29. Logaritmo"), id="op_logaritmo"),
                    ListItem(Label("30. Potência [c,gama]"), id="op_potencia"),
                    ListItem(Label("31. Fatiamento de bits [nome_saida]"), id="op_fatiamento"),
                    ListItem(Label("32. Histograma normalizado"), id="op_hist_normalizado"),
                    ListItem(Label("33. Histograma acumulado"), id="op_hist_acumulado"),
                    ListItem(Label("34. Hist. acumulado normalizado"), id="op_hist_acumulado_normalizado"),
                    ListItem(Label("35. Espectro de Fourier"), id="op_espectro_fourier"),
                    ListItem(Label("36. Gaussiano - Passa-baixa [d0]"), id="op_gaussiano_baixa"),
                    ListItem(Label("37. Gaussiano - Passa-alta [d0]"), id="op_gaussiano_alta"),
                    ListItem(Label("38. Rejeita-banda (máscara) [caminho_mascara]"), id="op_rejeita_mascara"),
                    ListItem(Label("39. Passa-banda [d0,W]"), id="op_banda_passa"),
                    ListItem(Label("40. Rejeita-banda [d0,W]"), id="op_banda_rejeita"),
                )
                yield Label(
                    "Como usar o parametro: (legenda?)\n"
                    "  tam/limiar → um número\n"
                    "  k-vizinhos → tam,k\n"
                    "  sementes → x,y\n"
                    "  região → x,y,limiar\n"
                    "  canny → limiar_baixo,limiar_alto,tam\n"
                    "  contraste → c,d\n"
                    "  potência → c,gama (aceita decimal)\n"
                    "  fatiamento → nome_saida (texto, não número)\n"
                    "  gaussiano → d0\n"
                    "  rejeita-banda (máscara) → caminho da imagem de máscara\n"
                    "  passa/rejeita-banda → d0,W\n"
                    "\n"
                    "Buffer para atalho",
                    classes="legenda",
                )
                yield Label("", id="digitando")
                yield Label("", id="status")

            with Vertical(id="area-principal"):
                yield ImagemTerminal(id="tela_render")

        yield Footer()

    #func para iniciar o estado da aplicação e guardar referencias para os campos
    def on_mount(self) -> None:
        self.tela = self.query_one("#tela_render", ImagemTerminal)
        self.status = self.query_one("#status", Label)
        self.digitando_label = self.query_one("#digitando", Label)
        self.input_param = self.query_one("#input_param", Input)
        self.input_img = self.query_one("#input_img", Input)
        self.caminho_imagem = self.input_img.value
        self.caminho_atual = self.caminho_imagem # ultima img gerada na aplicacao dos filtros

        self._buffer_numero = ""
        self._timer_numero = None

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input_img":
            self.caminho_imagem = event.value
            self.caminho_atual = event.value  #troca imagem

    #para ter atalho nos nums
    def on_key(self, event) -> None:
        if isinstance(self.focused, Input):
            return  #precisei colocar isso para evitar se caso eu estivesse em um campo de input, n ser interpretado como atalho

        tecla = event.key

        if tecla.isdigit():
            if self._timer_numero:
                self._timer_numero.stop()
                self._timer_numero = None

            self._buffer_numero += tecla

            # limitar pq nenhuma opção passa de 2 dígitos
            if len(self._buffer_numero) >= 2:
                self._confirmar_buffer()
            else: #se for só uma tecla (uma opção de so um numero, por exemplo: 5)
                self._atualizar_label_digitando()
                self._timer_numero = self.set_timer(TEMPO_ESPERA_SEGUNDO_DIGITO, self._confirmar_buffer)
            event.stop()

        elif tecla == "enter" and self._buffer_numero:
            if self._timer_numero:
                self._timer_numero.stop()
                self._timer_numero = None
            self._confirmar_buffer()
            event.stop()

        elif tecla == "escape" and self._buffer_numero:
            self._buffer_numero = ""
            self._atualizar_label_digitando()
            event.stop()

    # essa func mostra oq ta digitando
    def _atualizar_label_digitando(self) -> None:
        if self._buffer_numero:
            self.digitando_label.update(f"Digitando: {self._buffer_numero}")
        else:
            self.digitando_label.update("")

    # zerar buffer
    def _confirmar_buffer(self) -> None:
        numero_str = self._buffer_numero
        self._buffer_numero = ""
        self._timer_numero = None
        self._atualizar_label_digitando()

        if not numero_str:
            return

        numero = int(numero_str)
        if 0 <= numero < len(ATALHOS):
            lista = self.query_one(ListView)
            lista.index = numero
            # chama a exec da operacao
            self.executar_operacao(ATALHOS[numero])
        else:
            self.status.update(f"Opção {numero} não existe (use 0 a {len(ATALHOS) - 1})")

    # quando seleciona uma opção, executa a operacao
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.executar_operacao(event.item.id)

    def executar_operacao(self, opcao: str) -> None:
        # coloquei em um dicionario no wrapper pra diminuir o código na main
        handler = OPERACOES.get(opcao)
        if handler is None:
            return

        if opcao == "op_original":
            self.caminho_atual = self.caminho_imagem  # reseta a imagem par ao original

        try:
            resultado = handler(self.caminho_atual, self.input_param.value)
            if isinstance(resultado, PILImage.Image):
                self.tela.image = resultado
            else:
                self.tela.image = cv2_para_pil(resultado) #exibe a imagem na tela

            self.caminho_atual = salvar_como_atual(resultado) # salva a imagem com o ultimo filtro aplicado
            self.status.update("")
        except Exception as e:
            self.status.update(f"Erro na operação: {e}")


if __name__ == "__main__":
    app = TrabalhoPDI()
    app.run()
