import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts
from flask import Flask, request

## Load the mapping for Bot responses

with open("mapping.json", "r") as f:
    mapping = json.load(f)
    f.close()

## Read the data

df = pd.read_csv("training.csv", encoding="latin1")

print(df.head())

## Feature Extraction using the Bag of Words (BoW) approach

## Input Text
X = df["Query"]

## Output (Labels)
y = df["Intent"]

# Initialize the CountVectorizer

cv = CountVectorizer(ngram_range=(1,2))

X_tf = cv.fit_transform(X)

## Split the data into Train and Validation sets

X_train, X_test, y_train, y_test = tts(X_tf,y, test_size = 0.2, random_state=123)

## Check the dataframe shape

print(X_train.shape, y_train.shape)

## Fit the logistic regression model

model = LogisticRegression()

model.fit(X_train, y_train)

## Prediction using the trained model

print(clsr(y_test,model.predict(X_test)))

if __name__=="__main__":

    print("Bot> Hi, I am a Bot to help you manage your daily groceries. Please type to continue")
    while True:
        text = input()
        print("You> "+str(text))
        print("Bot> "+mapping[model.predict(cv.transform([text])[0])[0]])


