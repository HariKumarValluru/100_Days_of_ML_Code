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