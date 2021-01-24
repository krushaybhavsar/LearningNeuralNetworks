# LearningMachineLearning

## Overview
This is a repository to document my progress on learning basic machine learning concepts.

## Training Data
Fashion/Clothing Classification Tensorflow in-built training data (Fashion MNIST): tf.keras.datasets.fashion_mnist → https://www.tensorflow.org/tutorials/keras/classification
</br>
Text Classification/Sentinement Analysis training data (IMDB User Movie Reviews): https://ai.stanford.edu/~amaas/data/sentiment/
</br>
Linear Regression training data (UCI ML Repository): https://archive.ics.uci.edu/ml/datasets/Student+Performance
</br>
K-Nearest Neighbors training data (UCI ML Repository): https://archive.ics.uci.edu/ml/datasets/car+evaluation
</br>
Support Vector Machine sklearn in-built training data (Scikit Learn Dataset): datasets.load_breast_cancer() → https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## What I Learned
Below is a summary description of what I learned.
</br></br>
### Image Classification
##### Preparing Data
Image classification can be done usuing neural networks. In order to prepare the image data for classification, you can convert color images, if there are any, into black and white images. Each pixel of the black and white image will have a certain brightness level ranging from 0 to 255. Often times, you may want to work with smaller numbers, so it is better to later divide the brightness level by 255 to get a decimal between 0 and 1.
##### Model Architecture
Since the training data I used contained images that were 28 pixels by 28 pixels, the input layer can have 784 (28 times 28) neurons. I can add a hidden layer with an activation functions of relu to further optimize the network. Since there are 10 different outcomes the image can be, the output layer should have 10 neurons and I can use a softmax function. I made use of loss functions, which allows the computer to know how far the predicted outcome deviates from the actual outcome.
</br></br>
### Text Classification
##### Preparing Data
Simply explained, text classification works by mapping each word with to a numerical value. In order for the code to work for texts of different lengths, I can add tags such as UNUSED so that the computer knows to ignore those words. For example, if the number of words in a text is 700, and the arbitrary max number of words I had set was 1000, the computer will add 300 UNUSED tags to fill the rest of the spots up. Similar tags can be used for unknown words (UNK) and for padding characters (PAD).
##### Model Architecture
The neural network I used contained four layers. The input layer had 88000 neurons, which was the arbitrary number I chose for the max words. I has two hidden layers, one being a GlobalAveragePooling1D layer and another being a Dense layer. According to Tensorflow documentation, "GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible". The output layer has only 1 neuron. Using a sigmoid function as the activation function, the neuron will have a value that is either closer to 0 or 1 (0 being negative and 1 being positive). Based on this value of 0 to 1, the computer can differentiate between text with more negative words vs text with more positive words.
### Linear Regression
Description soon to be updated
### K-Nearest Neighbors
Description soon to be updated
### Support Vector Machine
Description soon to be updated
