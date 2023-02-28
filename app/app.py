from flask import Flask, render_template, request,jsonify
from model import *
import logging

#Création de l'API Flask et du server Flask
app = Flask(__name__)

#Route accueil v1
@app.route('/', methods=['GET'])
def index():
    return 'Retinal OCT prediction API'

# Route accueil v2
@app.route("/home")
def main():
    return render_template('index.html')

#Route prédiction v1
@app.route('/predict', methods=['POST'])
def infer_image():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    
    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return str(predict_result(img))
    
#Route prédiction v2
@app.route('/predictv2', methods=['POST'])
def predict_image_file():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    
    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    pred = predict_result(img)

    match pred:
        case 0:
            retine="Néovascularisation choroïdienne"
        case 1:
            retine="Oedème maculaire diabétique"
        case 2:
            retine="Multiples drusen"
        case 3:
            retine="Aucune anomalie"
    # Return on a JSON format
    return render_template("result.html", predictions=retine)

#Driver
if __name__ == '__main__':
    app.run(debug=True, port='5500')
