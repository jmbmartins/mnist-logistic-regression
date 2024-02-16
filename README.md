# Logistic Regression on MNIST Dataset

This repository contains the implementation of Logistic Regression for classifying handwritten digits from the MNIST dataset. The project includes a Jupyter Notebook (`Pratical_Project_02.ipynb`) with the code implementation, a report (`Relatorio_v1.1.pdf`) detailing the project, and a Python script (`logistic_regression.py`) for training and testing the model.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. It is split into 60,000 training examples and 10,000 test examples.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) in CSV format. The `mnist_train.csv` file contains training examples and labels, while `mnist_test.csv` contains test examples and labels.

## Features
- Normalization Strategies:
  - Min-Max Scaling
  - Z-score Standardization

## Model Considerations
- Regularization Values
- Stopping Criteria
- Learning Rates

## Implementation
The `logistic_regression.py` script preprocesses the data, trains the logistic regression model, and evaluates its performance on the test set. It includes functions for computing loss, sigmoid activation, feedforward operation, and model fitting.

## Usage
1. Run the Jupyter Notebook `Pratical_Project_02.ipynb` to see the code implementation.
2. Execute the `logistic_regression.py` script to train and test the model on the MNIST dataset.
3. Update the repository with any modifications or enhancements to the code.

