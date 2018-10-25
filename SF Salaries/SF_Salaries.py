# SF Salaries Dataset from Kaggle!

# Importing the libraries
import pandas as pd

# Importing the dataset
salary = pd.read_csv("Salaries.csv")

# Check the head of the DataFrame. 
salary.head()

salary.info()

# Calculating the average basepay
salary['BasePay'].mean()

# highest amount of OvertimePay
salary['OvertimePay'].max()

# job title of JOSEPH DRISCOLL 
salary[salary['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']

# JOSEPH DRISCOLL total benefits
salary[salary['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']

# highest paid person
salary.iloc[salary['TotalPayBenefits'].argmax]

# lowest paid person
salary.iloc[salary['TotalPayBenefits'].argmin]

# average (mean) BasePay of all employees per year
salary.groupby('Year').mean()['BasePay']

# unique job titles
salary['JobTitle'].nunique()

# top 5 most common jobs
salary['JobTitle'].value_counts().head()

# Job Titles were represented by only one person in 2013
sum(salary[salary['Year'] == 2013]['JobTitle'].value_counts() == 1)

# people have the word Chief in their job title
def chief_str(title):
    if "chief" in title.lower().split():
        return True
    else:
        return False
sum(salary['JobTitle'].apply(lambda x: chief_str(x)))

# correlation between length of the Job Title string and Salary
salary['title_len'] = salary['JobTitle'].apply(len)

salary[['TotalPayBenefits','title_len']].corr()