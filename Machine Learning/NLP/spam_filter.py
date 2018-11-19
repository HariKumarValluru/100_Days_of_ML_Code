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
    