import os
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

app = Flask(__name__)

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

train_data = data[:4400]

# Train model
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data.v2)

classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True))
classifier.fit(X_train, train_data.v1)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None

    if request.method == "POST":
        message = request.form["message"]
        vectorized_message = vectorizer.transform([message])
        prediction = classifier.predict(vectorized_message)[0]
        prob = classifier.predict_proba(vectorized_message).tolist()

    return render_template("index.html", prediction=prediction, prob=prob)

if __name__ == "__main__":
    app.run(debug=True)
