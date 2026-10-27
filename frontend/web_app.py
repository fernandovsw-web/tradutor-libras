"""
web_app.py
 
PARA QUE SERVE
--------------
É o servidor do site do projeto. Ele usa o Flask (uma biblioteca para criar
sites em Python) para entregar ao navegador as páginas do projeto: a página
inicial e a página do tradutor.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Cria a aplicação Flask.
2. Define duas "rotas", que são os endereços do site:
   - "/" (página inicial): entrega o arquivo index.html.
   - "/tradutor": entrega o arquivo tradutor.html.
3. Ao rodar o script, o servidor sobe no seu computador. Por padrão, o site
   fica disponível no navegador em http://127.0.0.1:5000
4. Quando alguém acessa um endereço, o Flask executa a função daquela rota,
   que devolve a página correspondente.
 
O QUE ESTE ARQUIVO NÃO FAZ
--------------------------
Ele só entrega as páginas. Não abre a webcam, não carrega o modelo e não
reconhece letras: nada disso acontece neste arquivo.
 
ESTRUTURA ESPERADA
------------------
O Flask procura os arquivos HTML dentro de uma pasta chamada "templates",
ao lado deste script. Então precisam existir:
    templates/index.html
    templates/tradutor.html
 
ENTRADAS
--------
- Os acessos feitos pelo navegador aos endereços "/" e "/tradutor".
 
SAÍDA
-----
- As páginas HTML mostradas no navegador.
 
DEPENDÊNCIAS
------------
flask.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from flask import Flask, render_template

# Cria a aplicação. O __name__ ajuda o Flask a descobrir onde estão a pasta
# "templates" e os outros arquivos do site
app = Flask(__name__)

# ---------------------------------------------------------------------------
# ROTAS (os endereços do site)
# ---------------------------------------------------------------------------

@app.route("/")
def inicio():
    # Página inicial: entrega o arquivo templates/index.html
    return render_template("index.html")


@app.route("/tradutor")
def tradutor():
    # Página do tradutor: entrega o arquivo templates/tradutor.html
    return render_template("tradutor.html")

# ---------------------------------------------------------------------------
# INICIALIZAÇÃO DO SERVIDOR
# ---------------------------------------------------------------------------

# Este bloco só roda quando o arquivo é executado diretamente
# (python web_app.py), e não quando ele é importado por outro arquivo.
if __name__ == "__main__":
    # debug=True: o servidor reinicia sozinho quando o código muda e mostra
    # os erros detalhados no navegador. É ótimo para desenvolver, mas não
    # deve ficar ligado quando o site for publicado de verdade.
    app.run(debug=True)
