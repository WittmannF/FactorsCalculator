# DS Utility Functions

## `ml.RegressorCV` Class

### Overview
The `RegressorCV` class in the `ml.py` module of the `ds_utils` library is designed to train an estimator using cross-validation and record metrics for each fold. It also stores each model that is trained on each fold of the cross-validation process. The final prediction is made as the median value of the predictions from each stored model.

### Initialization
```python
RegressorCV(base_reg, cv=5, groups=None, verbose=False, n_bins_stratify=None)
```

#### Parameters
- `base_reg` : Estimator object implementing 'fit'. The object to use to fit the data.
- `cv` : int or cross-validation generator, default=5. Determines the cross-validation splitting strategy.
- `groups` : array-like of shape (n_samples,), default=None. Group labels for the samples used while splitting the dataset into train/test set.
- `n_bins_stratify` : int, default=None. Number of bins to use for stratification.
- `verbose` : bool, default=False. Whether or not to print metrics.

### Attributes
- `cv_results_` : Dictionary containing the results of the cross-validation, including the fold number, the regressor, and the metrics.
- `oof_train_` : Series containing the out-of-fold predictions on the training set.
- `oof_score_` : The R2 score calculated using the out-of-fold predictions.
- `oof_mape_` : The Mean Absolute Percentage Error calculated using the out-of-fold predictions.
- `oof_rmse_` : The Root Mean Squared Error calculated using the out-of-fold predictions.
- `metrics_` : The summary of metrics calculated during the fitting process.

### Methods

#### fit(self, X, y, **kwargs)
Trains the base regressor on the provided data using cross-validation and stores the results.

#### predict(self, X)
Predicts the target variable for the provided features using the median value of the predictions from the models trained during cross-validation.

### Example
```python
from ds_utils.ml import RegressorCV
from sklearn.ensemble import RandomForestRegressor

# Initialize the RegressorCV with a base regressor and cross-validation strategy
reg_cv = RegressorCV(base_reg=RandomForestRegressor(), cv=5, verbose=True)

# Fit the RegressorCV to the training data
reg_cv.fit(X_train, y_train)

# Predict the target variable for new data
predictions = reg_cv.predict(X_new)

# Get the summary of recorded metrics
metrics = reg_cv.metrics_
```

## `ml.RegressorTimeSeriesCV` Class

### Overview
The `RegressorTimeSeriesCV` class in the `ml.py` module of the `ds_utils` library is designed to train a base regressor using time series cross-validation and record metrics for each fold.

### Initialization
```python
RegressorTimeSeriesCV(base_reg, cv=5, verbose=False, catboost_use_eval_set=False)
```

#### Parameters
- `base_reg` : Estimator object implementing 'fit'. The object to use to fit the data.
- `cv` : int, default=5. Determines the cross-validation splitting strategy.
- `verbose` : bool, default=False. Whether or not to print metrics.
- `catboost_use_eval_set` : bool, default=False. Whether or not to use eval_set in CatBoostRegressor.

### Attributes
- `cv_results_` : List containing the results of the cross-validation, including fold number, regressor, train and test indices, and metrics.
- `metrics_` : The summary of metrics calculated during the fitting process.
- `y_test_last_fold_` : The true target variable values of the last fold.
- `y_pred_last_fold_` : The predicted target variable values of the last fold.

### Methods

#### fit(self, X, y, sample_weight=None)
Trains the base regressor on the provided data using time series cross-validation and stores the results.

#### predict(self, X)
Predicts the target variable for the provided features using the base regressor trained on the full data.

### Example
```python
from ds_utils.ml import RegressorTimeSeriesCV
from sklearn.ensemble import RandomForestRegressor

# Initialize the RegressorTimeSeriesCV with a base regressor and cross-validation strategy
reg_tscv = RegressorTimeSeriesCV(base_reg=RandomForestRegressor(), cv=5, verbose=True)

# Fit the RegressorTimeSeriesCV to the training data
reg_tscv.fit(X_train, y_train)

# Predict the target variable for new data
predictions = reg_tscv.predict(X_new)

# Get the summary of recorded metrics
metrics = reg_tscv.metrics_
```

## `ml.KNNRegressor` Class

### Overview
The `KNNRegressor` class in the `ml.py` module of the `ds_utils` library is an extension of the `KNeighborsRegressor` from scikit-learn, with modifications to the `predict` method to allow different calculations for predictions and to optionally return the indices of the nearest neighbors.

### Initialization
The initialization parameters are the same as those of the `KNeighborsRegressor` from scikit-learn. Refer to the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) for details on the parameters.

### Methods

#### predict(self, X, return_match_index=False, pred_calc='mean')
Predicts the target variable for the provided features and allows different calculations for predictions.

##### Parameters
- `X` : array-like of shape (n_samples, n_features). Test samples.
- `return_match_index` : bool, default=False. Whether to return the index of the nearest matched neighbor.
- `pred_calc` : str, default='mean'. The calculation to use for predictions. Possible values are 'mean' and 'median'.

##### Returns
- `y_pred` : array of shape (n_samples,) or (n_samples, n_outputs). The predicted target variable.
- `nearest_matched_index` : array of shape (n_samples,). The index of the nearest matched neighbor. Returned only if `return_match_index=True`.
- `neigh_ind` : array of shape (n_samples, n_neighbors). Indices of the neighbors in the training set. Returned only if `return_match_index=True`.

### Example
```python
from ds_utils.ml import KNNRegressor

# Initialize the KNNRegressor with specific parameters
knn_reg = KNNRegressor(n_neighbors=3)

# Fit the KNNRegressor to the training data
knn_reg.fit(X_train, y_train)

# Predict the target variable for new data and return the index of the nearest matched neighbor
predictions, nearest_matched_index, neigh_ind = knn_reg.predict(X_new, return_match_index=True, pred_calc='median')
```

## `ml.AutoRegressor` Class

### Overview
The `AutoRegressor` class is designed for performing automated regression tasks, including preprocessing and model fitting. It supports several regression algorithms and allows for easy comparison of their performance on a given dataset. The class provides various methods for model evaluation, feature importance, and visualization.

### Initialization
```python
ar = AutoRegressor(
    num_cols,
    cat_cols,
    target_col,
    data=None,
    train=None,
    test=None,
    random_st=42,
    log_target=False,
    estimator='catboost',
    imputer_strategy='simple',
    use_catboost_native_cat_features=False,
    ohe_min_freq=0.05,
    scale_numeric_data=False,
    scale_categoric_data=False,
    scale_target=False
)
```

#### Parameters
- **num_cols**: list
  - List of numerical columns in the dataset.
- **cat_cols**: list
  - List of categorical columns in the dataset.
- **target_col**: str
  - Target column name in the dataset.
- **data**: pd.DataFrame, optional (default=None)
  - Input dataset containing both features and target column.
- **train**: pd.DataFrame, optional (default=None)
  - Training dataset containing both features and target column. Used if `data` is not provided.
- **test**: pd.DataFrame, optional (default=None)
  - Testing dataset containing both features and target column. Used if `data` is not provided.
- **random_st**: int, optional (default=42)
  - Random state for reproducibility.
- **log_target**: bool, optional (default=False)
  - If the logarithm of the target variable should be used.
- **estimator**: str or estimator object, optional (default='catboost')
  - String or estimator object. Options are 'catboost', 'random_forest', and 'linear'.
- **imputer_strategy**: str, optional (default='simple')
  - Imputation strategy for missing values. Options are 'simple' and 'knn'.
- **use_catboost_native_cat_features**: bool, optional (default=False)
  - If the native CatBoost categorical feature handling should be used.
- **ohe_min_freq**: float, optional (default=0.05)
  - Minimum frequency for OneHotEncoder to consider a category in categorical columns.
- **scale_numeric_data**: bool, optional (default=False)
  - If numeric data should be scaled using StandardScaler.
- **scale_categoric_data**: bool, optional (default=False)
  - If categorical data (after one-hot encoding) should be scaled using StandardScaler.
- **scale_target**: bool, optional (default=False)
  - If the target variable should be scaled using StandardScaler.

### Methods

#### fit_report(self)
Fits the model to the training data and prints the R2 Score, RMSE, and MAPE on the test data.

#### test_binary_column(self, binary_column)
Tests the significance of a binary column on the target variable and returns the p-value for Mann–Whitney U test.

#### get_coefficients(self)
Returns the coefficients of the model if the estimator is linear, otherwise returns feature importances.

#### get_feature_importances(self)
Returns the feature importances of the model.

#### get_shap(self, return_shap_values=False)
Generates and plots SHAP values for the model and returns SHAP values if `return_shap_values` is True.

#### plot_importance(self, feat_imp, graph_title="Model feature importance")
Plots the feature importances provided in `feat_imp` with the specified `graph_title`.

### Example Usage
```python
# Initialize AutoRegressor
ar = AutoRegressor(num_cols, cat_cols, target_col, data)

# Fit the model and print the report
ar.fit_report()

# Get and plot feature importances
feat_imp = ar.get_feature_importances()
ar.plot_importance(feat_imp)
```