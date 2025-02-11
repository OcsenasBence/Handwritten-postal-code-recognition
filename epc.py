#-------------------------------------------------------------------
#Importing necessary libraries and modules.
import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Preparing the MNIST dataset for training a machine learning model.
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Defining a convolutional neural network (CNN) model using the Keras library.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Training a CNN model on the MNIST dataset for 20 epochs,
#evaluating its performance on the test set using validation data,
#and then saving the trained model to a file for future use.
hist = model.fit(train_X, train_y,
                epochs=20,
                validation_data=(test_X, test_y),
                callbacks=[early_stopping]
                )
model.save('mnist_cnn.keras')
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Converting the input image to grayscale, applying Gaussian blur, thresholding,
#morphological closing, and then normalizing the resulting binary image before returning it.
def normalize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = thresh.astype('float32') / 255.0
    return thresh
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Taking an input image, processes it to extract contours using various image processing techniques like thresholding and morphological operations,
#and then returns the contours found in the image.
def get_all_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Removing the top 1/3, extracting contours from the image.
image = cv2.imread('kep1.jpg')
height, width, _ = image.shape
image = image[height//3:height, :]
contours = get_all_contours(image)
digits = []
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Processing each contour by extracting a region of interest,
#preprocessing it, predicting the digit using a trained model,
#and filtering based on certain conditions before storing the results in a list.
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
#-------------------------------------------------------------------


#-------------------------------------------------------------------
#Sorting and processing the list of digit information to extract the top 4 digits with the highest accuracy,
#arrange them in left to right order basod on their positions in the image,
#and print them as a string aling with their average accuracy.
digits = sorted(digits, key=lambda x: x['accuracy'], reverse=True)[:4]
avg_accuracy = np.mean([d['accuracy'] for d in digits])
digits = sorted(digits, key=lambda x: cv2.boundingRect(x['cont'])[0])
extracted_digits = [d['digit'] for d in digits]
dig = ''.join([str(d) for d in extracted_digits])
print(f"Extracted digits: {dig} ({round(avg_accuracy*100, 2)}% accuracy)")
#-------------------------------------------------------------------