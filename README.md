# ResNet-50-Image-Classifier-Cats-vs.-Dogs
This repository contains the implementation of my project focused on image classification using the ResNet-50 architecture. The goal is to classify images of cats and dogs into two categories while evaluating model performance under different conditions, including clean, noisy, and combined datasets. The project utilizes transfer learning, data augmentation, and a GUI-based testing interface.

## Requirements
This repository is tested on Windows 11 and requires:
- Python 3.12
- CUDA 12.4
- PyTorch 2.5.1

### Development Environment
- PyCharm Community Edition 2024.3.1.1

## Getting Started

### 1. Install the necessary libraries:
- torch, torchvision, numpy, matplotlib, scikit-learn, PIL, tkinter, tensorboard
### 2. Data Preparation
The dataset is divided into three categories:

- Clean dataset: Standard images of cats and dogs.
- Noisy dataset: Clean images with added Gaussian noise.
- Combined dataset: A mix of clean and noisy images.

Images are resized to **224x224** pixels for compatibility with ResNet-50.

### 3. Adding Noise to Images
To generate the noisy dataset, use the **add_noise.py** script! This script applies Gaussian noise to clean images and saves them in a separate folder.

### 4. Training the Model

The training script main.py is used to train the **ResNet-50** model with different datasets. The model can be trained on clean, noisy and combined datasets.

Modify these options in main.py:
```bash
use_noise_dataset = False  # Set to True to train on the noisy dataset
use_combined_dataset = True  # Set to True to train on the combined dataset
```

### 5. Training Configuration

- Optimizer: SGD with momentum
- Loss function: Cross-Entropy Loss
- Learning rate: 0.001
- Batch size: 100
- Epochs: 10

TensorBoard is used for tracking training progress. To visualize:
```bash
tensorboard --logdir .\logs
```
![image](https://github.com/user-attachments/assets/4a987af2-84bd-471b-9bc0-7b8a29d4c1cf)

### 6. Model Testing

A GUI-based testing application is provided in **test_app.py**.
![image](https://github.com/user-attachments/assets/bcbcf9c5-b16c-4d73-8d62-7910f9ddcd7a)
![image](https://github.com/user-attachments/assets/b12ed3a2-dace-4b64-a0b0-a01f6273b7e7)

This app allows the user to:
- Select an image for classification.
- Choose between models trained on different datasets.
- Apply noise to test images before classification.

### 7. Results and Analysis
