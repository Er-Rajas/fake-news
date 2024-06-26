from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    text = data['text']
    prediction = predict(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
