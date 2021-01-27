# Learning Machine Learning

## Overview
This is a repository to document my progress in learning basic machine learning concepts.

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
### Image Classification
##### Preparing Data
Image classification can be done using neural networks. In order to prepare the image data for classification, you can convert color images, if there are any, into black and white images. Each pixel of the black and white image will have a certain brightness level ranging from 0 to 255. Often times, you may want to work with smaller numbers, so it is better to later divide the brightness level by 255 to get a decimal between 0 and 1.
##### Model Architecture
Since the training data I used contained images that were 28 pixels by 28 pixels, the input layer can have 784 (28 times 28) neurons. I can add a hidden layer with an activation function of relu to further optimize the network. Since there are 10 different outcomes the image can be, the output layer should have 10 neurons and I can use a softmax function. I made use of loss functions, which allows the computer to know how far the predicted outcome deviates from the actual outcome.
</br></br>
### Text Classification
##### Preparing Data
Simply explained, text classification works by mapping each word with to a numerical value. In order for the code to work for texts of different lengths, I can add tags such as UNUSED so that the computer knows to ignore those words. For example, if the number of words in a text is 700, and the arbitrary max number of words I had set was 1000, the computer will add 300 UNUSED tags to fill the rest of the spots up. Similar tags can be used for unknown words (UNK) and padding characters (PAD).
##### Model Architecture
The neural network I used contained four layers. The input layer had 88000 neurons, which was the arbitrary number I chose for the max words. I had two hidden layers, one being a GlobalAveragePooling1D layer and another being a Dense layer. According to Tensorflow documentation, "GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible". The output layer has only 1 neuron. Using a sigmoid function as the activation function, the neuron will have a value that is either closer to 0 or 1 (0 being negative and 1 being positive). Based on this value of 0 to 1, the computer can differentiate between text with more negative words vs text with more positive words.
</br></br>
### Linear Regression
Linear Regression can be used to predict outcomes based on previous data. The computer essentially plots each data point on a graph (could be 2D or even 3D depending on how many parameters you have that are affecting the outcome). From the plotted points, the computer can calculate a line of best fit. Using the line of best fit and basic data trends, the computer is able to predict an outcome. Linear regression, however, assumes that data is linearly correlated and may not always be as accurate. Sometimes, data can follow a curve, rather than a straight line. You can change the degree of the line by making it quadratic, cubic, quartic, etc. depending on what suits your data best. Aside from just the linear regression aspect of this project, I learned how I can save models without having to re-train them every time I run them. This can be done using the pickle module, which will save the data from the model into a pickle file. Later, I can simply open and load the file into the model. Matplotlib is also a helpful tool to visualize your data.
</br></br>
### K-Nearest Neighbors
K-Nearest Neighbors is an algorithm that can be used to predict the class a data point associates with. When data tends to clump into groups and you want to predict which group a certain point is a part of, KNN comes in handy. KNN locates the closest specified number of points within a certain range to a data point. Depending on the points around it, the computer can predict which group the unknown point belongs in. For the data I worked with, finding the 9 closest neighbors worked the best for predicting unknown points. It is important to tweak this number depending on how your data is.
</br></br>
### Support Vector Machine
Support Vector Machine is an algorithm that splits data points into different sides using planes. Although there is multiclass SVM, for my project, I only dealt with the simple SVM algorithm. Because of this, my data can be split into either of two cases. For my project, I used breast cancer data and predicted whether a cancer is malignant or benign. Once all of the data points are plotted, the computer draws a plane in a position such that the distance between the two points closest to the line from either side of the line is the same. Data usually varies and separating the points with a hard plane may not give your algorithm a high accuracy. Therefore, sometimes it is important to insert a "padding" so that it allows some points to be on the opposite side of the line. I manually set this value and tweaked it to give me better accuracy. After successfully completing the goal of the project, I compared the accuracy of the SVM algorithm with the accuracy of the KNN algorithm. Shockingly, KNN performed better than SVM on this particular dataset, with KNN having an accuracy of about 98.25% and SVM having 96.49%. The accuracy of the SVM varied more as well. Each time a ran the code, SVM gave me an accuracy between 89% to 97%. KNN varied slightly from about 97% to 99%. Based on these results, I can only conclude that KNN worked better than SVM on this particular dataset, but cannot say it will consistently work better on all data. It is also important to note that I may not have optimized some settings, such as padding, with SVM.
</br></br>
### K-Means Clustering
K-Means Clustering works by plotting a K number of centroids in a pool of data points. In my case, the centroids were initially placed in random positions. The computer then finds the distance from each point to either of the centroids. If one point is closer to a certain centroid, it will mark it as part of that class. It will do this for all the points until each of the points is marked as part of a class out of K number of classes. After this, the computer calculates the average location of the data points in a specific class. The centroid that belongs to that class will then be repositioned to that calculated location. This happens for each of the centroids. The computer then checks the surrounding data points again and reassigns them to the class of the centroid they are closest to. This will happen over and over again until no points are reassigned to a different class. At this stage, the computer has successfully distinguished data points into a K number of classes. When an unknown data point is given, the computer is able to predict what class the data point belongs to. K-Means Clustering, however, has a downside when it comes to large data sets since it has to continuously calculate and recalculate distances for each of the data points. The dataset I used predicted what digits were being shown in data from handwritten digits. Since there are 10 different digits, K was equal to 10 in my case.
</br></br>
## Credits
Mostly all of this was learned by watching [Tech with Tim's Youtube Channel](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg). For more information on [machine learning algorithms](https://www.youtube.com/watch?v=ujTCoH21GlA&list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr) and [neural networks](https://www.youtube.com/watch?v=OS0Ddkle0o4&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj), I highly recommend his playlists.
