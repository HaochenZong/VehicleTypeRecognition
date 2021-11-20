# VehicleTypeRecognition

### Classification task
Categorization of images is a classical machine learning task, which has been a key driver in machine learning research since the birth of deep neural networks. In this project, I learned to classify images of different vehicle types, including cars, bicycles, vans, ambulances, etc. (total 17 categories).

### The data
The data has been collected from the Open Images dataset; an annotated collection of over 9 million images. We are using a subset of openimages, selected to contain only vehicle categories among the total of 600 object classes.

### Methods
1. Method1: using scikit-learn tools and pretrained convnet
    - I used scikit-learn tools with a pretrained convnet for
feature extraction. This is not the most usual way to use convnets, but can be a good approach if the amount of data is too small for proper training of convnet. 
    - I created a convnet, pass all images through it, fit a sklearn model and predict the classes for test data.
    - I tried different classifiers: Linear discriminant analysis classifier, Support vector machine (linear kernel), Support vector machine (RBF kernel), Logistic regression, Random forest and finally compared the results.

2. Method2: train a full deep learning model 
    -  Trained a convolutional network with Tensorflow / Keras, such as Mobilenet, MobilenetV2 and InceptionV3.
    - To improve the performance, I also used data augmentation and other network architecture, such as EfficientNet.
