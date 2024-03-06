import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from datePreprocess import df_all

# Split the dataset into train and test sets
train_set, combine_set = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
test_set, val_set = train_test_split(combine_set, test_size=0.5, random_state=42, stratify=combine_set['label'])

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_set['text'])
X_test = vectorizer.transform(test_set['text'])
X_val = vectorizer.transform(val_set['text'])

# Convert labels into numerical values
y_train = train_set['label']
y_test = test_set['label']
y_val = val_set['label']

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict the labels for validation set
y_pred = svm_model.predict(X_val)

# Calculate the accuracy of the model
accuracy_val = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy_val)


# Predict emotions for test samples
for i in range(3):
    text = val_set['text'].to_list()[i]
    features = vectorizer.transform([text])
    prediction = svm_model.predict(features)
    print('Prediction:', text)
    print('Predicted:', prediction)
    print('RealLabel:', val_set['label'].to_list()[i])
