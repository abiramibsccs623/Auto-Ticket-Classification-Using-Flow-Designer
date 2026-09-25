from flask import Flask, render_template, request
from utils.prediction import predict_ticket

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    category = None
    ticket = ""
    if request.method == "POST":
        ticket = request.form.get("ticket", "")
        category = predict_ticket(ticket)
    return render_template("index.html", category=category, ticket=ticket)

if __name__ == "__main__":
    app.run(debug=True)
