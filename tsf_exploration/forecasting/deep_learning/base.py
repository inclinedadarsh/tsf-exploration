"""Base class for deep learning forecasters."""

__all__ = ["BaseDeepForecaster"]

from abc import abstractmethod

import tensorflow as tf

from aeon.forecasting.base import BaseForecaster


class BaseDeepForecaster(BaseForecaster):
    """Abstract base class for deep learning time series forecasters.

    The base deep forecaster provides a deep learning default method for
    _fit and _predict, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    horizon : int, default=1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead.
    batch_size : int, default=32
        Training batch size for the model
    last_file_name : str, default="last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used
    random_state : int, default=None
        Random seed for reproducibility
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def __init__(
        self,
        horizon=1,
        batch_size=32,
        last_file_name="last_model",
        random_state=None,
        axis=0,
    ):
        self.batch_size = batch_size
        self.last_file_name = last_file_name
        self.random_state = random_state
        self.model_ = None
        self.history = None

        super().__init__(horizon=horizon, axis=axis)

    @abstractmethod
    def build_model(self, input_shape):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def _fit(self, y, exog=None):
        """Fit the deep learning model to the training data.

        Parameters
        ----------
        y : np.ndarray
            Training time series data
        exog : np.ndarray, default=None
            Optional exogenous variables (not supported yet)

        Returns
        -------
        self : BaseDeepForecaster
            Reference to self.
        """
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")

        # Build the model if not already built
        if self.model_ is None:
            self.model_ = self.build_model(input_shape=y.shape)

        # Set random seed for reproducibility
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)

        # Train the model
        self.history = self.model_.fit(
            y,
            y,  # For now, we're using the same data as input and target
            batch_size=self.batch_size,
            epochs=1,  # This should be a parameter in child classes
            verbose=0,
        )

        return self

    def _predict(self, y=None, exog=None):
        """Make predictions using the trained model.

        Parameters
        ----------
        y : np.ndarray, default=None
            Time series data to predict from
        exog : np.ndarray, default=None
            Optional exogenous variables (not supported yet)

        Returns
        -------
        predictions : np.ndarray
            Forecasted values
        """
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")

        if y is None:
            raise ValueError("y cannot be None for prediction")

        # Make predictions
        predictions = self.model_.predict(y)
        return predictions

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default="./"
            The directory where the model will be saved

        Returns
        -------
        None
        """
        self.model_.save(file_path + self.last_file_name + ".keras")

    def load_model(self, model_path):
        """Load a pre-trained keras model instead of fitting.

        Parameters
        ----------
        model_path : str
            Path to the saved model file

        Returns
        -------
        None
        """
        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

    def summary(self):
        """Get the training history of the model.

        Returns
        -------
        history : dict or None
            Dictionary containing model's training history
        """
        return self.history.history if self.history is not None else None
