from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("HDI.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Prediction page
@app.route("/Prediction")
def prediction():
    return render_template("indexnew.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():

    # Get input values from the form
    life = float(request.form["life"])
    school = float(request.form["school"])
    gni = float(request.form["gni"])
    internet = float(request.form["internet"])

    # Create DataFrame with the same columns used for training
    data = pd.DataFrame(
        [[life, school, gni, internet]],
        columns=[
            "Life expectancy",
            "Mean years of schooling",
            "Gross national income (GNI) per capita",
            "Internet users"
        ]
    )

    # Predict HDI
    prediction = model.predict(data)[0]

    # Display the result
    return render_template(
        "resultnew.html",
        prediction=round(prediction, 3)
    )

if __name__ == "__main__":
    app.run(debug=True)