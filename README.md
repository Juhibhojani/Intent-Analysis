# Intent-Analysis
Tweet Intent Analysis using Natural Language Processing

# Overview
<br>
<b>Task</b> : Performing Intent Analysis on tweets<br>
<b> Class Labels</b>: Appreciation, Community, Done, Giveaway, Interested, Launching Soon, PinkSale, PreSale, Whitelist


# Steps Performed
1) Importing Necessary libraries<br>
2) Performing Exploratory Data Analysis
3) Data Cleaning<br>
4) Tokenization 
5) Stemming and Lemming
6) Displaying Word Cloud
7) Applying the Tf-Idf vectorizer and different models
8) Applying Tf-Idf with bigrams 
9) Predicting Output Class
10) Pickling


# Models
1) SGD Classifier
2) Random Forest Classifier
3) Logistic Regression
4) KNN
5) Naive Bayes
6) SVM

# REST API using Flask
Using pickle files and Flask, a request-response framework is set up.

# Result<br>
<br>
Using the Tf-Idf vectorizer for feature extraction, we obtained the highest accuracy of 0.91 using the Random Forest Classifier.

# Scope of Improvement
1) Using a layered approach to initially classify into 4 classes i.e. Community, Giveaway, Appreciation, and Others. In the second step, classify other class labels into remaining unbalanced classes i.e. Presale, Whitelist,pinksale, Done, Interested, Launching Soon.
2) Using Word2Vec as a word embedding technique.

