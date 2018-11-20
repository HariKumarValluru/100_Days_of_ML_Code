# Spam Filter
import nltk

#nltk.download()

# reading messages from the file
messages = [line.rstrip() for line in open("Datasets/SMS_Spam_Collection.csv")]

print(len(messages))

# checking one item
print(messages[0])

for i, message in enumerate(messages[:10]):
    print(i, message + "\n")

import pandas as pd

messages = pd.read_csv("Datasets/SMS_Spam_Collection.csv", sep="\t", 
                       names=['label', 'message'])

messages.head()

messages.describe()

messages.groupby('label').describe()

# new column length of message
messages['length'] = messages['message'].apply(len)

messages.head()

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=70)

messages['length'].describe()

messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=60, figsize=(12,5))

import string

from nltk.corpus import stopwords

# checking list of stop words
stopwords.words('english')

def txt_process(message):
    """
    1. remove punctuation
    2. remove stop words
    3. return list of clean text words
    """
    no_punc = [char for char in message if char not in string.punctuation]
    
    no_punc = "".join(no_punc)
    
    return [word for word in no_punc.split() if word.lower not in 
            stopwords.words('english')]
    
messages['message'].head(5).apply(txt_process)

# implementing bag of words
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=txt_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]

print(mess4)

bow4 = bow_transformer.transform([mess4])

print(bow4)

print(bow4.shape)

bow_transformer.get_feature_names()[9832]
    
messages_bow = bow_transformer.transform(messages['message'])

# number of non zeros
messages_bow.nnz

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

# Sparsity
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print("Sparsity: {}".format(sparsity))

# checking the document frequency of a word
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

spam_detect_model.predict(tfidf4)[0]

messages['label'][3]

all_pred = spam_detect_model.predict(messages_tfidf)

# splitting the messages to train and test sets
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(
        messages['message'], messages['label'], test_size=0.3)

# creating a pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=txt_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        ])
    
pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print(classification_report(label_test, predictions))