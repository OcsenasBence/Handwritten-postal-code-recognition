from keras.models import load_model
import sys
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def normalize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = thresh.astype('float32') / 255.0
    return thresh

def get_all_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

model = load_model('mnist_cnn.keras')

image_path = sys.argv[1]
image = cv2.imread(image_path)
height, width, _ = image.shape
image = image[height//3:height, :]
contours = get_all_contours(image)

digits = []
for index, cont in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cont)
    box = image[y:y + h, x:x + w]
    average_blue_color = np.mean(box[:, :, 0])
    normalized = normalize_image(box)
    pad = 20
    padded = cv2.copyMakeBorder(normalized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    resized = cv2.resize(padded, (28, 28))
    resized.astype('float32')
    score = model.predict(resized.reshape(-1, 28, 28, 1))
    if np.max(score) > 0.75 and average_blue_color > 170:
        t = {
            'img': resized,
            'digit': np.argmax(score),
            'accuracy': np.max(score),
            'cont': cont
        }
        digits.append(t)
digits = sorted(digits, key=lambda x: x['accuracy'], reverse=True)[:4]
avg_accuracy = np.mean([d['accuracy'] for d in digits])
digits = sorted(digits, key=lambda x: cv2.boundingRect(x['cont'])[0])
extracted_digits = [d['digit'] for d in digits]

dig = ''.join([str(d) for d in extracted_digits])
print(f"Extracted digits: {dig} ({round(avg_accuracy*100, 2)}% accuracy)")
