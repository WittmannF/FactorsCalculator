from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier # KNeighborsRegressor,
try:
    from .ml import KNNRegressor
except ImportError:
    from ml import KNNRegressor
from catboost import CatBoostRegressor
import numpy as np

class TreeBasedKNeighbors(BaseEstimator):
    """
    Base class for combining a Tree-Based Model with a K-Nearest Neighbors model.
    The tree-based model is trained first, and the feature importances from it 
    are then used as weights for the KNN model.
    """
    def __init__(self, k=5, weights='uniform', tree_model=None):
        """
        Initialize the model with given hyperparameters.
        Parameters:
        - k: The number of nearest neighbors to consider in the KNN model.
        - weights: Weight function to use in prediction for the KNN model. Possible values: 'uniform', 'distance'.
        - tree_model: The tree-based model to use. If None, a default RandomForest instance will be used.
        """
        self.k = k
        self.tree_model = tree_model
        self.knn_model = None
        self.weights = weights
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        To be implemented by subclasses.
        """
        pass

    def transform(self, X):
        """
        Transform the input using the feature importances from the tree-based model.
        Parameters:
        - X: The input data.
        """
        # If X is a scipy sparse matrix, convert it to a dense array
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        transformed_X = X * self.feature_importances_
        
        return transformed_X

    def predict(self, X):
        """
        Transform the input using the feature importances, then pass it to the KNN model for prediction.
        Parameters:
        - X: The input data.
        """
        transformed_X = self.transform(X)
        return self.knn_model.predict(transformed_X)


class TreeBasedKNeighborsRegressor(TreeBasedKNeighbors, RegressorMixin):
    """
    A regressor that combines a Tree-Based Model with a K-Nearest Neighbors model.
    """
    def fit(self, X, y):
        """
        Fit the model to the input data.
        Parameters:
        - X: The input features.
        - y: The target values.
        """
        if self.tree_model is None:
            self.tree_model = RandomForestRegressor()  # Default to RandomForestRegressor if no estimator is provided

        # Train tree-based model
        self.tree_model.fit(X, y)

        # Get feature importances
        self.feature_importances_ = self.tree_model.feature_importances_

        # Transform input data using feature importances
        transformed_X = self.transform(X)

        self.knn_model = KNNRegressor(n_neighbors=self.k, weights=self.weights)
        self.knn_model.fit(transformed_X, y)


class TreeBasedKNeighborsClassifier(TreeBasedKNeighbors, ClassifierMixin):
    """
    A classifier that combines a Tree-Based Model with a K-Nearest Neighbors model.
    """
    def fit(self, X, y):
        """
        Fit the model to the input data.
        Parameters:
        - X: The input features.
        - y: The target labels.
        """
        if self.tree_model is None:
            self.tree_model = RandomForestClassifier()  # Default to RandomForestClassifier if no estimator is provided

        # Train tree-based model
        self.tree_model.fit(X, y)

        # Get feature importances
        self.feature_importances_ = self.tree_model.feature_importances_

        # Transform input data using feature importances
        transformed_X = self.transform(X)

        self.knn_model = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
        self.knn_model.fit(transformed_X, y)

    def predict_proba(self, X):
        """
        Return class probabilities for each input.
        Parameters:
        - X: The input data.
        """
        transformed_X = self.transform(X)
        return self.knn_model.predict_proba(transformed_X)

if __name__ == "__main__":
    # Load the Ames Housing dataset
    housing = fetch_openml(name="house_prices", as_frame=True)
    X = housing.data
    y = housing.target

    # Define preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=object))])

    # Preprocessing
    X = preprocessor.fit_transform(X)

    # Standardize the target
    y = y.values.reshape(-1, 1)
    scaler_y = StandardScaler().fit(y)
    y = scaler_y.transform(y).ravel()  # Convert back to 1d array

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('TreeBasedKNeighborsRegressor', TreeBasedKNeighborsRegressor()),
        ('TreeBasedKNeighborsRegressor with Distance Weights', TreeBasedKNeighborsRegressor(weights='distance')),
        ('TreeBasedKNeighborsRegressor with Distance Weights and CatBoost', TreeBasedKNeighborsRegressor(weights='distance', tree_model=CatBoostRegressor(verbose=0))),
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(random_state=42)),
        ('KNN', KNNRegressor(weights='uniform')),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f'{name} Train Score: {train_score}\n{name} Test Score: {test_score}\n')
        # Output:
        # TreeBasedKNeighborsRegressor Train Score: 0.8639150318097282
        # TreeBasedKNeighborsRegressor Test Score: 0.8462399719884531
        # 
        # TreeBasedKNeighborsRegressor with Distance Weights Train Score: 1.0
        # TreeBasedKNeighborsRegressor with Distance Weights Test Score: 0.8691328950107591
        # 
        # TreeBasedKNeighborsRegressor with Distance Weights and CatBoost Train Score: 1.0
        # TreeBasedKNeighborsRegressor with Distance Weights and CatBoost Test Score: 0.878429926102895
        # 
        # Linear Regression Train Score: 0.9401416030865126
        # Linear Regression Test Score: 0.4443195721134888
        # 
        # Random Forest Train Score: 0.9795564593459206
        # Random Forest Test Score: 0.8919694549777907
        # 
        # KNN Train Score: 0.8337200992316732
        # KNN Test Score: 0.8068127103988856