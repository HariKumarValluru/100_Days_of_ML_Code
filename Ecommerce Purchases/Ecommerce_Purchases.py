# Importing the libraries and dataset
import pandas as pd

ecom = pd.read_csv('Ecommerce Purchases.csv')

# checking dataframe
ecom.head()

ecom.info()

# calculating the average purchase price
ecom['Purchase Price'].mean()

# calculating the highest and lowest prices
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()

# English 'en' as their Language of choice on the website
ecom[ecom['Language'] == 'en'].count()

# Job title of "Lawyer"
ecom[ecom['Job'] == 'Lawyer'].info()

# purchase during the AM and how many people made the purchase during PM
ecom['AM or PM'].value_counts()

# 5 most common Job Titles
ecom['Job'].value_counts().head(5)

# purchase that came from Lot: "90 WT"
ecom[ecom['Lot'] == '90 WT']['Purchase Price']

# email of the person
ecom[ecom['Credit Card'] == 4926535242672853]['Email']

# people have American Express as their Credit Card Provider *and made a purchase above $95 
ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count()

# credit cards that expires in 2025?
sum(ecom['CC Exp Date'].apply(lambda exp: exp[3:] == '25'))
ecom[ecom['CC Exp Date'].apply(lambda exp: exp[3:] == '25')].count()

# top 5 most popular email providers/hosts
ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().head(5)
