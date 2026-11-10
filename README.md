# Convolutional Neural Networks (CNN)
This project implements a convolutional neural network using PyTorch with four classical machine learning methods implemented in comparison. We will evaluate all five models on the four image classification datasets. The main goal is to study how model performance changes with image complexity, color and amount of training data. 

# Datasets

| Dataset | Images | Image size | Classes | Purpose |
| --- | ---: | --- | ---: | --- |
| [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) | 70,000 | 28 × 28, grayscale | 10 | Recognize handwritten digits and establish a baseline on clean, centered shapes. |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 70,000 | 28 × 28, grayscale | 10 | Recognize clothing items while keeping the image dimensions and dataset size the same as MNIST. |
| [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) | 60,000 | 32 × 32, RGB | 10 | Recognize objects with color, background clutter, and variation within each class. |
| [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | 5,640 | Variable, approximately 300–640 pixels per dimension, RGB | 47 | Recognize texture patterns with limited training examples per class. |

There is roughly 2-3GB for the extracted data. All four datasets are available through torchvision.datasets: MNIST, FashionMNIST, CIFAR10, and DTD.

# Comparison
CNN will learn through the convolutional layers and other methods. We will compare them to the following methods;

| Method | Description |
| --- | --- |
| Logistic regression | A linear classifier that provides a simple baseline. |
| K-nearest neighbors (KNN) | Predicts a class using nearby training examples in feature space. |
| Random forest | Combines predictions from multiple classification trees. |
| Support vector machine (SVM) | Learns class boundaries using a linear or kernel-based decision function. |

We will use the same training, validation, and test set for every method that is implemented. 

# Result

# How to run

Install python3 and then run the following commands in the root.

```python
python3 -m venv .venv
source .venv/bin/activate

# make sure that pip is install and updated to newest
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# downloading required packages for this project
python -m pip install -r requirements.txt
```

If requirements is already installed, just activate .venv when starting a new terminal

# References
- Slides: https://docs.google.com/presentation/d/1wnJMYiNNhWZ4wx8B--sQabTAN9TiTCJwYPzcYqq-NJE/edit?usp=sharing
- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto technical report.
- Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. (2014). Describing Textures in the Wild. CVPR.

