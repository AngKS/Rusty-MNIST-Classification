# Rusty: Image Classification using MNIST Dataset
<!-- add gif image -->
[training]()

## Introduction
This project is a Rust implementation of a simple neural network that classifies images from the MNIST dataset. The neural network is a simple feedforward neural network with 2 Convolutional layers, 1 adaptive Pooling layer, and 2 hidden layers. The network is trained using the backpropagation algorithm. The project is implemented using the Rust programming language.

## Model Architecture
The model architecture is as follows:

1. Convolutional 2D Layer
2. Convolutional 2D Layer
3. Adaptive Average Pooling Layer
4. Dropout Layer
5. Fully Connected Layer
6. Fully Connected Layer
7. ReLU Activation Function

## Training
The training process configuration is as follows:

```bash
Optimizer: Adam
Number of Epochs: 5
Batch Size: 64
Number of Workers: 4
Learning Rate: 0.00001
```

## Results
The model achieved an accuracy of 97.72% on the test dataset.

```bash
======================== Learner Summary ========================
Model: Model[num_params=531178]
Total Epochs: 5


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 82.872   | 1        | 96.160   | 5        |
| Train | Loss     | 0.125    | 5        | 0.605    | 1        |
| Valid | Accuracy | 93.430   | 1        | 97.720   | 5        |
| Valid | Loss     | 0.070    | 5        | 0.230    | 1        |
```
