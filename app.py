# app.py

import os
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

# Serve images from /models/ directory
@app.route('/models/<path:filename>')
def model_static(filename):
    return send_from_directory('models', filename)

@app.route("/")
def dashboard():
    # Hardcoded accuracy for now — replace with real values if needed
    accuracy = {
        "Prophet": "95.3%",
        "XGBoost": "94.1%",
        "LSTM": "93.7%"
    }

    return render_template("index.html", accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
