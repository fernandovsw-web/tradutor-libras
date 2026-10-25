
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/tradutor")
def tradutor():
    return render_template("tradutor.html")


if __name__ == "__main__":
    app.run(debug=True)
