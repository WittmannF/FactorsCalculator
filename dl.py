#from sentence_transformers import SentenceTransformer, util
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier#KNeighborsRegressor, KNeighborsClassifier
from .ml import KNNRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np

class NeuralKNeighbors(BaseEstimator):
    """
    Base class for combining a Neural Network with a K-Nearest Neighbors model.

    The neural network is trained first, and the features from its last hidden layer 
    are then used as input for the KNN model.
    """
    def __init__(
        self, hidden_layer_sizes, k=5, weights='uniform', verbose=0, epochs=50, batch_size=32, 
        use_last_layer_weights=False
    ):
        """
        Initialize the model with given hyperparameters.

        Parameters:
        - hidden_layer_sizes: List of integers indicating the number of neurons in each hidden layer of the neural network.
        - k: The number of nearest neighbors to consider in the KNN model.
        - weights: Weight function to use in prediction for the KNN model. Possible values: 'uniform', 'distance'.
        - verbose: Verbosity mode for neural network training.
        - epochs: Number of epochs for neural network training.
        - batch_size: Batch size for neural network training.
        - use_last_layer_weights: Whether to multiply the output of the last hidden layer by the weights of the output layer.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.k = k
        self.nn_model = None
        self.knn_model = None
        self.weights = weights
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_last_layer_weights = use_last_layer_weights

    def _build_nn_model(self, input_dim, output_dim, output_activation, loss):
        """
        Build and compile the neural network model.

        Parameters:
        - input_dim: Number of input features.
        - output_dim: Number of output features.
        - output_activation: Activation function for the output layer.
        - loss: Loss function to use for training.
        """
        model = Sequential()
        model.add(Dense(self.hidden_layer_sizes[0], input_dim=input_dim, activation='relu'))

        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))

        model.add(Dense(output_dim, activation=output_activation))
        model.compile(loss=loss, optimizer='adam')
        return model

    def fit(self, X, y):
        """
        To be implemented by subclasses.
        """
        pass

    def transform(self, X):
        """
        Transform the input using the neural network and return the output of the last hidden layer.

        Parameters:
        - X: The input data.
        """
        transformed_X = self.feature_model.predict(X, verbose=self.verbose)
        if self.use_last_layer_weights:
            last_layer_weights = self.nn_model.layers[-1].get_weights()[0]
            # Reshape the weights to be a 1D array
            last_layer_weights = last_layer_weights.reshape(-1)
            transformed_X = transformed_X * last_layer_weights
        return transformed_X

    def predict(self, X):
        """
        Transform the input using the neural network, then pass it to the KNN model for prediction.

        Parameters:
        - X: The input data.
        """
        transformed_X = self.transform(X)
        return self.knn_model.predict(transformed_X)


class NeuralKNeighborsRegressor(NeuralKNeighbors, RegressorMixin):
    """
    A regressor that combines a Neural Network with a K-Nearest Neighbors model.
    """
    def fit(self, X, y):
        """
        Fit the model to the input data.

        Parameters:
        - X: The input features.
        - y: The target values.
        """
        # Train neural network
        self.nn_model = self._build_nn_model(X.shape[1], 1, output_activation='linear', loss='mean_squared_error')
        self.nn_model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # Extract embeddings (features from last hidden layer)
        embeddings = self.nn_model.layers[-2].output
        self.feature_model = keras.Model(inputs=self.nn_model.input, outputs=embeddings)

        # Transform input data using neural network
        transformed_X = self.transform(X)

        self.knn_model = KNNRegressor(n_neighbors=self.k, weights=self.weights)
        self.knn_model.fit(transformed_X, y)


class NeuralKNeighborsClassifier(NeuralKNeighborsRegressor, ClassifierMixin):
    """
    A classifier that combines a Neural Network with a K-Nearest Neighbors model.
    NOTE: Not tested yet.
    """
    def fit(self, X, y):
        """
        Fit the model to the input data.

        Parameters:
        - X: The input features.
        - y: The target labels.
        """
        num_classes = len(np.unique(y))
        self.nn_model = self._build_nn_model(X.shape[1], num_classes, output_activation='softmax', loss='sparse_categorical_crossentropy')
        self.nn_model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # Extract embeddings (features from last hidden layer)
        embeddings = self.nn_model.layers[-2].output
        self.feature_model = keras.Model(inputs=self.nn_model.input, outputs=embeddings)
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


"""
class ZeroShotTextClassifier:
    def __init__(self, language='pt', model_ref=None):
        if model_ref is not None:
            self.model = SentenceTransformer(model_ref)
        elif language=='en':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif language=='pt':
            self.model = SentenceTransformer('rufimelo/bert-large-portuguese-cased-sts')
    
    def predict(self, X, labels, return_type='labels'):
        embedding_1= self.model.encode(X, convert_to_tensor=True)
        embedding_2= self.model.encode(labels, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
        idxs, percents = cos_sim.max(axis=1).indices, cos_sim.max(axis=1).values
        if return_type=='labels_with_percent':
            text_labels = []
            for i, percent in zip(idxs, percents):
                tl = f'{labels[i]} ({percent*100:.2f}%)'
                text_labels.append(tl)
            return text_labels
        if return_type=='labels':
            pred_labels = [labels[i] for i in idxs]
            return pred_labels
        else:
            return idxs, percents

"""