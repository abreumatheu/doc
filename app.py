from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
from twilio.rest import Client
import os

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Configuração do Twilio
TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'
DEST_WHATSAPP_NUMBER = 'whatsapp:+5511943471217'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_photo', methods=['POST'])
def send_photo():
    file = request.files['photo']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    message = client.messages.create(
        body='Olá, estou enviando a foto do meu documento.',
        from_=TWILIO_WHATSAPP_NUMBER,
        to=DEST_WHATSAPP_NUMBER,
        media_url=f'http://your-server-url/{file_path}'
    )

    return jsonify({'status': 'sucesso', 'message_sid': message.sid})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
