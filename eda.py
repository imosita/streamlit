from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Basic info
print(df.info())
print(df.describe())

# Check class distribution
print(df['target'].value_counts())  # 0 = malignant, 1 = benign

# Visualize class balance
sns.countplot(x='target', data=df)
plt.title('Benign vs Malignant Tumor Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Distribution of key features
sns.histplot(df['mean radius'], kde=True)
plt.title('Distribution of Mean Radius')
plt.show()   