# Fashion MNIST Classification using CNN

This project is a Convolutional Neural Network (CNN) implementation to classify images from the **Fashion MNIST dataset**. The dataset consists of 70,000 grayscale images of 10 different categories of clothing items.

## Project Overview
The goal of this project is to build a CNN model that can accurately classify clothing items into one of the following categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Features
- **Image Classification**: Uses CNN to classify images from the Fashion MNIST dataset.
- **Deep Learning Model**: Leverages multiple convolutional layers with max pooling, followed by fully connected layers.
- **Training & Validation**: Tracks accuracy and loss on both training and validation data.

## Dataset
- **Fashion MNIST**: A dataset of **28x28 grayscale images** representing different clothing items, with **10 classes**.
- **Size**: 60,000 training images and 10,000 test images.
- **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

## Model Architecture
1. **Convolutional Layers**: Extract features from input images.
2. **Max Pooling**: Reduces spatial dimensions and computation.
3. **Flattening**: Converts 2D feature maps into a 1D vector.
4. **Dense (Fully Connected) Layers**: Perform classification based on extracted features.
5. **Softmax Output**: Provides a probability distribution over the 10 clothing categories.

## Installation
### Prerequisites
Ensure you have Python installed. Recommended version: **Python 3.7+**

### Clone the Repository
```bash
git clone https://github.com/TheGravityFalls-11/MNIST_img_classification_CNN.git
cd MNIST_img_classification_CNN
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Model
To train and test the CNN model, run the following command:
```bash
python train.py
```

Or, if using Jupyter Notebook:
```bash
jupyter notebook
# Open model.ipynb and run all cells
```

## Usage
- Train the model using the Fashion MNIST dataset.
- Evaluate performance on test data.
- Modify hyperparameters and experiment with different architectures.

## Results
The CNN model achieves high accuracy on Fashion MNIST, making it a great benchmark for image classification tasks.

## License
This project is open-source and available under the [MIT License](LICENSE).


