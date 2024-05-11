import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load pre-trained face cascade and facial expression recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
expression_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Define route for processing user input
@app.route('/process', methods=['POST'])
def process():
    text_input = request.form['text_input']
    image_file = request.files['image_file']

    # Process image for face detection
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform face detection using OpenCV
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces are detected
    face_detected = len(faces) > 0

    # Perform facial expression recognition on detected faces
    expressions = []
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for each face
        face_roi = gray_image[y:y + h, x:x + w]
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)

        # Perform facial expression recognition on the ROI
        expression_model.setInput(face_blob)
        detections = expression_model.forward()
        max_index = np.argmax(detections[0])
        expression_label = expression_labels[max_index]
        expressions.append(expression_label)

    # Perform natural language processing on text input
    sentiment_score = sia.polarity_scores(text_input)

    # Return processed data as JSON response
    return jsonify({
        'sentiment_score': sentiment_score,
        'face_detected': face_detected,
        'expressions': expressions
    })

if __name__ == '__main__':
    app.run(debug=True)