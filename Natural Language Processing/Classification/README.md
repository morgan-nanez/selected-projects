# Sentiment Analysis with Logistic Regression

## Overview

I will implement logistic regression and apply it to a sentiment analysis dataset.

## Data Preprocessing

 
### Feature Extraction
I used an Indexer which is used for creating a mapping between words and indices. 
The Indexer class serves as a valuable tool for establishing a bijective relationship between objects and integers, commencing at zero. This functionality proves particularly advantageous when needing to map labels, features, or other entities into the coordinates of a vector space. The class achieves this by creating a dual mapping system—associating each unique object (in this context, words) with a distinctive index and vice versa. For instance, the mapping might assign the index 1 to "apple" and 2 to "banana." The class provides methods for retrieving objects based on their indices, checking the presence of an object, determining the index of a given object, and dynamically adding objects to the index with a corresponding nonnegative index value. Overall, the Indexer class offers a versatile and efficient means of managing the correlation between objects and integers within a computational context.

In addition to the indexer, I used both Unigram and Bigram Features and compare how use these different extractors affect model accuracy.

An example of input code is as follows:

```python
['there', 'is', 'a', 'freedom', 'to', 'watching', 'stunts', 'that', 'are', 'this', 'crude', ',', 'this', 'fast-paced', 'and', 'this', 'insane', '.']; label=1 # label = 1 is a positive review
```

There are 6920 train examples: 3610 positive, 3310 negative. There are 872 testing examples.



### Inital Classifer
I use a basic Logistic Regression model, that I created myself. Manuelly, I wrote the Stochastic Gradient Desenct method, as well as. It takes a feature extractor (unigram or bigram), training examples, and hyperparameters such as the number of iterations, regularization parameters, and learning rate. The class initializes the logistic regression model by defining variables for weights and biases and calls the train function to optimize the model using gradient descent. 

The train function iterates through the dataset multiple times, applying the update rule to weights and biases in each step. The predict function computes the logistic regression model's prediction for a single example, utilizing the extracted features and the sigmoid function. I trained for 50 epochs.

For this first iteration. I did not use an regularization.

| Feature Extractor | Trainin Accuracy| Testing Accuracy |
| --------------- | --------------- |---------------|
|Unigram| 94.41%|77.29%|
|Bigram| 97.41%|64.68%|


### With Regularization

Without regularization, we can see from above that training accuracies are much higher, while testing accuracy reamins low. This leads me to believe that there is over fitting.

I tested seven different regularization parameters, ranging from 10^-5 to 1