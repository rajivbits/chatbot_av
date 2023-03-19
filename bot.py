import re
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

## Read mapping for bot response

with open("message.json", "r") as f:
    response = json.load(f)
    f.close()

## Read the data

data = pd.read_csv("training.csv")

print(data.columns)

def preprocess(text):

    return re.sub(r'^[a-z-0-9]', " ", text.lower())

## Feature selection

cv = CountVectorizer(ngram_range=(1,3))
labels = MultiLabelBinarizer()

X_vect = cv.fit_transform([preprocess(x) for x in data['Query']])
y_label = data['Intent']

## Train Model

model = LogisticRegression()

model.fit(X_vect,y_label)


if __name__=="__main__":

    print("Neo: Hi, My Name is Neo, I can help you manage your subscription. How can I help you today?")
            
    while True:
        
        t = input()
        print("You: {}".format(t))
        print("Neo: {}".format(response[model.predict(cv.transform([t]))[0]]))
        
        

        