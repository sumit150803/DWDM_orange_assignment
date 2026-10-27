from sklearn.datasets import load_iris, load_diabetes
import pandas as pd
import numpy as np

# 1. iris.arff
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
df_iris['class'] = [iris.target_names[t] for t in iris.target]
class_map = {'setosa': 'Iris-setosa', 'versicolor': 'Iris-versicolor', 'virginica': 'Iris-virginica'}

arff_lines = [
    '@relation iris',
    '',
    '@attribute sepallength numeric',
    '@attribute sepalwidth numeric',
    '@attribute petallength numeric',
    '@attribute petalwidth numeric',
    '@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}',
    '',
    '@data'
]
for _, row in df_iris.iterrows():
    c_name = class_map[row['class']]
    arff_lines.append(f"{row['sepallength']},{row['sepalwidth']},{row['petallength']},{row['petalwidth']},{c_name}")

with open('iris.arff', 'w') as f:
    f.write('\n'.join(arff_lines) + '\n')
print("Created iris.arff")

# 2. Dataset for Data Cleaning (with missing values, noisy outliers, inconsistent formats)
np.random.seed(42)
n_samples = 200
ages = np.random.randint(18, 75, size=n_samples).astype(float)
incomes = np.random.normal(55000, 15000, size=n_samples).round(2)
credit_scores = np.random.randint(300, 850, size=n_samples).astype(float)
departments = np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], size=n_samples)
loan_approved = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.65, 0.35])

# Introduce missing values
ages[np.random.choice(n_samples, 18, replace=False)] = np.nan
incomes[np.random.choice(n_samples, 15, replace=False)] = np.nan
credit_scores[np.random.choice(n_samples, 12, replace=False)] = np.nan

df_clean = pd.DataFrame({
    'Age': ages,
    'Annual_Income': incomes,
    'Credit_Score': credit_scores,
    'Department': departments,
    'Loan_Approved': loan_approved
})
df_clean.to_csv('customer_loan_dirty.csv', index=False)
print("Created customer_loan_dirty.csv")

# 3. Market basket dataset for Association Rules
baskets = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Diaper', 'Beer'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Eggs', 'Butter'],
    ['Bread', 'Milk', 'Butter', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Milk', 'Bread', 'Jam'],
    ['Beer', 'Diaper', 'Chips'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Diaper', 'Beer', 'Cola'],
    ['Bread', 'Butter', 'Coffee'],
    ['Milk', 'Coffee', 'Bread']
]
with open('supermarket_basket.basket', 'w') as f:
    for b in baskets:
        f.write(', '.join(b) + '\n')
print("Created supermarket_basket.basket")
