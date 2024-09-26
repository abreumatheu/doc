from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data['image'].split(',')[1]  # Remover cabeçalho 'data:image/png;base64,...'
    image_bytes = base64.b64decode(image_data)

    # Converter bytes para imagem OpenCV
    image = Image.open(BytesIO(image_bytes))
    image_np = np.array(image)

    # Processar imagem (detecção de nitidez e qualidade)
    document_type = detect_document_type(image_np)
    quality_check = check_image_quality(image_np)

    return jsonify({'document_type': document_type, 'quality': quality_check})

def detect_document_type(img):
    # Usar OCR para verificar se é RG ou CNH
    text = pytesseract.image_to_string(img)
    if "CNH" in text:
        return "CNH"
    elif "RG" in text:
        return "RG"
    else:
        return "Desconhecido"

def check_image_quality(img):
    # Verificar nitidez (variação da Laplaciana para detectar embaçamento)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:
        return "Imagem embaçada"
    else:
        return "Imagem clara"

if __name__ == '__main__':
    app.run(debug=True)
