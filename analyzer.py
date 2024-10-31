import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
file_path = 'CollegeDistance.csv'
data = pd.read_csv(file_path)

# Wyświetlenie informacji o danych
data.info()
print(data.describe())

sns.set(style="whitegrid")

# Rozkłady dla zmiennych liczbowych
num_cols = ['score', 'unemp', 'wage', 'distance', 'tuition', 'education']
fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, 20))

for i, col in enumerate(num_cols):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Rozkłady zmiennych kategorycznych
cat_cols = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']
fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 20))

for i, col in enumerate(cat_cols):
    sns.countplot(data=data, x=col, hue=col, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()