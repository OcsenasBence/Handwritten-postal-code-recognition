---

# Handwritten Postal Code Recognition

This project was developed during my university studies in the "Image Processing and Medical Imaging" course. The goal was to create a program capable of recognizing handwritten postal codes.

## Project Overview

The project focuses on processing images containing handwritten postal codes and utilizing machine learning techniques to accurately interpret the digits.

## Repository Contents

- `epc.py`: The main script that processes input images and performs postal code recognition.
- `extract.py`: A utility script designed to extract and preprocess individual digit images from larger images containing postal codes.
- `mnist_cnn.keras`: A pre-trained Convolutional Neural Network (CNN) model tailored for digit recognition tasks.
- `digit.model`: The serialized model file used for digit classification.
- `kep1.jpg`, `kep2.jpg`, `kep3.jpg`: Sample images demonstrating the system's recognition capabilities.

## Getting Started

To run the project, ensure you have Python installed along with the necessary libraries. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/OcsenasBence/Handwritten-postal-code-recognition.git
   cd Handwritten-postal-code-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Recognition Script**:
   ```bash
   python epc.py path_to_image.jpg
   ```
   Replace `path_to_image.jpg` with the path to the image containing the handwritten postal code you wish to recognize.

## Model Training

The digit recognition model was trained using the MNIST dataset, which comprises a vast collection of handwritten digit images. A Convolutional Neural Network (CNN) was employed to achieve high accuracy in digit classification.

## Results

The system demonstrates reliable performance in recognizing handwritten postal codes, as illustrated in the sample images (`kep1.jpg`, `kep2.jpg`, `kep3.jpg`).

## Acknowledgments

This project was inspired by the foundational work in handwritten digit recognition, such as Yann LeCun's 1989 paper "Backpropagation Applied to Handwritten Zip Code Recognition". citeturn0search0

---
