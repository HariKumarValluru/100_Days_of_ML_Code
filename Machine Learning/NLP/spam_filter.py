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
    
    