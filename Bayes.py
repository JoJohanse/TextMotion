import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from datePreprocess import df_all

# Split the dataset into train and test sets
train_set, test_set = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])

# Create the count vectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_set['text'])

# Transform the test data
X_test = vectorizer.transform(test_set['text'])

# Create the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train, train_set['label'])

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(test_set['label'], predictions)
print('Accuracy:', accuracy)
for i in range(3):
    text = test_set['text'].to_list()[i]
    features = vectorizer.transform([text])
    prediction = classifier.predict(features)
    print('Prediction:', text)
    print('Predicted:', prediction)
    print('RealLabel:', test_set['label'].to_list()[i])