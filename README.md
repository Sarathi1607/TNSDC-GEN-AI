# Image Recognition using Convolutional Neural Network (CNN) with TensorFlow

## Introduction
This project implements image recognition using a Convolutional Neural Network (CNN) with TensorFlow. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Installation
1. Install Python 3.x from [python.org](https://www.python.org/).
2. Install TensorFlow using pip:
   ```
   pip install tensorflow
   ```
3. Install required dependencies:
   ```
   pip install numpy matplotlib
   ```

## Usage
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/your-repository.git
   ```
2. Run the script:
   ```
   python image_recognition_cnn.py
   ```


## Dataset
- **CIFAR-10**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is commonly used for image recognition tasks.

## Implementation Details
- **Model Architecture**: The CNN model consists of convolutional layers followed by max-pooling layers and fully connected layers.
- **Data Preprocessing**: The CIFAR-10 images are loaded and normalized to the range [0, 1].
- **Training**: The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.
- **Evaluation**: The trained model is evaluated on the test set to measure accuracy.

## Results
- The model achieves a certain accuracy on the test set after training for a specified number of epochs.
- Example images from the dataset are displayed with their corresponding predicted classes.
