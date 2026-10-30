# Tradutor-Libras
Repositório do GitHub dedicado ao projeto extensionista da disciplina de Processamento de Imagens. 
É um software feito em Python (versão 3.11.9) que traduz gestos feitos em Libras para texto.
Utiliza das bibliotecas externas OpenCV, MediaPipe, Joblib, Scikit-learn, Flask e Numpy para identificar os sinais feitos e traduzí-los para texto. 

**O programa não possui fluência em Libras**

Para a execução do projeto deve ser primeiramente instalado Python na versão mencionada acima, carregar o projeto e utilizar os seguintes comandos no terminal para instalar bibliotecas externas(note que aspas e vírgulas separam os comandos e não fazem parte deles): "pip install opencv-python", "pip install mediapipe", "pip install joblib", "pip install -U scikit-learn", "pip install virtualenv", "pip install Flask", "pip install numpy".

Após esta configuração para executar o projeto em si utilize o seguinte comando no terminal: "python -m frontend.web_app"
