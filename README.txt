Python 3.10 ou mais novo.
 
Bibliotecas Python:
  opencv-python-headless
  numpy
  matplotlib
  pillow
  pygame
  textual
  textual-image
 
Arquivos extras: -> tem que estar na mesma pasta do trabalho
  haarcascade_frontalface_default.xml
  audio.mp3

1) sudo apt update
2) sudo apt install python3 python3-pip python3-venv
3) cd pasta_do_trabalho
4) python3 -m venv venv
5) source venv/bin/activate
6) pip install opencv-python-headless numpy matplotlib pillow pygame textual textual-image
7) Se a camera der erro de permissao: sudo usermod -aG video $USER (precisa deslogar e logar de novo pra funcionar)

python3 main.py

COMO USAR
---------
- Campo "Caminho da Imagem": caminho de um arquivo
- Campo "Parametro": parametro da operacao selecionada (a legenda na tela mostra o formato esperado de cada uma)
- Lista de operacoes: navega com as setas e Enter, ou digita o numero da operacao
- Operacao "0. Original" reseta a imagem para o original
- Tecla "c": liga/desliga a camera
- Tecla "d": liga/desliga deteccao de rosto + musica (so funciona com camera ligada)

OBS: A imagem e exibida usando graficos reais do terminal (protocolo do Kitty ou Sixel), nao blocos de texto. Funciona bem em: Kitty, WezTerm, foot, Konsole (Sixel habilitado). Em terminais sem suporte (GNOME Terminal padrao), a imagem ainda aparece, so que como blocos de caracter.



