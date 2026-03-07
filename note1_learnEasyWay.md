# Learn ML the Easy Way — Iris Flower Classification

> Last updated: 2026-03-07 11:42
>
> Step-by-step guide based on: https://www.kaggle.com/code/agilesifaka/step-by-step-iris-ml-project/notebook

---

## 0) Setup

1. Download `Iris.csv` from Kaggle: https://www.kaggle.com/datasets/uciml/iris (also available here)
2. Put it in the `input/` folder:

```
prj1/
├── input/
│   └── Iris.csv
└── step-by-step-iris-ml-project.ipynb
```

---

## 1) Load the Dataset

```python
import numpy as np
import pandas as pd

iris_data = pd.read_csv('input/Iris.csv')
iris_data.head()
```

---

## 2) Quick Summary

```python
# check for missing data
print(iris_data.isnull().sum())

# drop Id column (not useful for training)
iris_data = iris_data.drop(['Id'], axis=1)

# shape
print("Dimension:", iris_data.shape)

# stats
print(iris_data.describe())

# class distribution
print(iris_data.groupby('Species').size())
```

---

## 3) Visualize

```python
import matplotlib.pyplot as plt
import seaborn as sns

# bar chart of class distribution
iris_data['Species'].value_counts().plot.bar(title='Flower class distribution')
plt.show()

# box plot
iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histogram
iris_data.hist()
plt.show()

# pairwise scatter plot
sns.pairplot(iris_data, hue="Species")
plt.show()
```

---

## 4) Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = iris_data.drop(['Species'], axis=1)
Y = iris_data['Species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
print("Train:", X_train.shape, "Test:", X_test.shape)
```

---

## 5) Train Models with Cross Validation

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = [
    ('LR',   LogisticRegression(solver='liblinear', multi_class='auto')),
    ('LDA',  LinearDiscriminantAnalysis()),
    ('CART', DecisionTreeClassifier()),
    ('KNN',  KNeighborsClassifier()),
    ('GNB',  GaussianNB()),
    ('SVC',  SVC(gamma='auto')),
]

names = []
accuracy = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    names.append(name)
    accuracy.append(cv_results)
    print(f"{name}: accuracy={cv_results.mean():.4f} std=({cv_results.std():.4f})")
```

---

## 6) Compare Models

```python
fig, ax = plt.subplots()
ax.boxplot(accuracy, labels=names)
ax.set_title('Model Accuracy Comparison')
plt.show()
```

---

## 7) Test Best Models

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

test_models = [
    ('KNN', KNeighborsClassifier()),
    ('GNB', GaussianNB()),
    ('SVC', SVC(gamma='auto')),
]

for name, model in test_models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(Y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
```

---

## 8) Conclusion


| Model | Type       | Notes                                      |
| ------- | ------------ | -------------------------------------------- |
| LR    | Linear     | Logistic Regression                        |
| LDA   | Linear     | Linear Discriminant Analysis               |
| CART  | Non-linear | Decision Tree                              |
| KNN   | Non-linear | K-Nearest Neighbors                        |
| GNB   | Non-linear | Gaussian Naive Bayes                       |
| SVC   | Non-linear | Support Vector Classifier — best accuracy |

SVC typically wins on this dataset with ~93% test accuracy.
