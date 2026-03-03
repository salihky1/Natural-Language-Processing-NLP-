import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup
import re

from sklearn.model_selection import train_test_split
data = pd.read_csv("NLPlabeledData.tsv", delimiter="\t", quoting=3)

sample_text = data.review[0]

sample_text = BeautifulSoup(sample_text).get_text()
sample_text = re.sub("[^a-zA-Z]", " ", sample_text)
sample_text = sample_text.lower()
sample_text = sample_text.split()

def process(review):
    review = BeautifulSoup(review).get_text()
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    stop_words = set(stopwords.words("english"))
    review = [w for w in review if w not in stop_words]
    return (" ".join(review))

train_x_all = []
for i in range(len(data["review"])):
    if (i + 1) % 1000 == 0:
        print("Processing step", i + 1)
    train_x_all.append(process(data["review"][i]))

x = train_x_all
y = np.array(data["sentiment"])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=33)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
train_x1 = vectorizer.fit_transform(train_x)

train_x1 = train_x1.toarray()
train_y1 = train_y

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_x1, train_y1)

test_x1 = vectorizer.transform(test_x)
test_x1 = test_x1.toarray()

from sklearn.metrics import roc_auc_score
test_predict = rf.predict(test_x1)
accuracy = roc_auc_score(test_y, test_predict)

sample_comment = "i am feeling tired"

sample_comment = process(sample_comment)
sample_comment = vectorizer.transform([sample_comment])
sample_comment = sample_comment.toarray()

prediction = rf.predict(sample_comment)

if prediction == 1:
    print("This comment: Positive")
else:
    print("This comment: Negative")

print(accuracy)
