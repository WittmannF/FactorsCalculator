# Associated Factors Calculator
Python library for quick calculation of associated factors for a given categorical variable using Logistic Regression

## Usage
```python
import pandas as pd
URL_DATA = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(URL_DATA)

# Factorize binary column to avoid collinearity
data['is_female'] = data.Sex.apply(lambda x: 1 if x=='female' else 0)

# Adding 'bad' features on purpose to reflect in the respective impact factors
numerical_cols = ['Age', 'Fare', 'PassengerId', 'SibSp', 'Parch', 'is_female', 'Pclass']
categorical_cols = ['Embarked', 'Name', 'Ticket', 'Cabin']
target_col = 'Survived'

fc = FactorsCalculator(
    data,
    numerical_cols,
    categorical_cols,
    target_col,
    factorize_target=False, # Set to true when target column has strings and need to be converted to int
    random_st=None # Run multiple times to check effect of random split 
)

```

Fit the estimator with:
```python
fc.fit_report()
```
Output:
```
Score: 0.8116591928251121
              precision    recall  f1-score   support

           0       0.84      0.85      0.84       133
           1       0.77      0.76      0.76        90

    accuracy                           0.81       223
   macro avg       0.80      0.80      0.80       223
weighted avg       0.81      0.81      0.81       223
```

Next, the normalized coefficients can be visualized with:
```python
# Valid for Logistic Regression
coef = fc.get_coefficients()
print(coef)
```
Output:
```
                           Survived
is_female                  1.298177
Pclass                    -0.689800
Age                       -0.436563
SibSp                     -0.247249
Cabin_missing_value       -0.151839
Cabin_infrequent_sklearn   0.151839
Parch                     -0.126036
Embarked_S                -0.087513
Embarked_Q                 0.074555
PassengerId                0.065413
Embarked_C                 0.045928
Fare                      -0.012896
Name_infrequent_sklearn    0.000000
Ticket_infrequent_sklearn  0.000000
```
