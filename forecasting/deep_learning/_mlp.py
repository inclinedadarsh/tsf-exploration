"""MLP-based time series forecaster."""

__all__ = ["MLPForecaster"]

import numpy as np
import tensorflow as tf

from forecasting.deep_learning.base import BaseDeepForecaster
from aeon.networks import MLPNetwork


class MLPForecaster(BaseDeepForecaster):
    """Multi-layer perceptron (MLP) based time series forecaster.

    Parameters
    ----------
    horizon : int, default=1
        The number of time steps ahead to forecast
    window_size : int, default=10
        The size of the sliding window used for training
    n_layers : int, default=3
        The number of dense layers in the MLP
    n_units : int or list of int, default=500
        Number of units in each dense layer
    activation : str or list of str, default='relu'
        Activation function(s) for each dense layer
    dropout_rate : float or list of float, default=None
        Dropout rate(s) for each dense layer
    dropout_last : float, default=0.3
        The dropout rate of the last layer
    use_bias : bool, default=True
        Whether to use bias values for dense layers
    batch_size : int, default=32
        Training batch size
    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        horizon=1,
        window_size=10,
        n_layers=3,
        n_units=500,
        activation="relu",
        dropout_rate=None,
        dropout_last=0.3,
        use_bias=True,
        batch_size=32,
        random_state=None,
        axis=0,
    ):
        super().__init__(
            horizon=horizon,
            batch_size=batch_size,
            random_state=random_state,
            axis=axis,
        )

        self.window_size = window_size
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_last = dropout_last
        self.use_bias = use_bias

        self._network = MLPNetwork(
            n_layers=n_layers,
            n_units=n_units,
            activation=activation,
            dropout_rate=dropout_rate,
            dropout_last=dropout_last,
            use_bias=use_bias,
        )

    def _prepare_data(self, y):
        """Prepare input and target data using sliding windows.

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        input_data : np.ndarray
            Input data with shape (n_samples, window_size, n_channels)
        target : np.ndarray
            Target data with shape (n_samples, horizon, n_channels)
        """
        n_samples = len(y) - self.window_size - self.horizon + 1
        n_channels = y.shape[1] if y.ndim > 1 else 1

        # Reshape y to 2D if univariate
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        input_data = np.zeros((n_samples, self.window_size, n_channels))
        target = np.zeros((n_samples, self.horizon, n_channels))

        for i in range(n_samples):
            input_data[i] = y[i : i + self.window_size]
            target[i] = y[i + self.window_size : i + self.window_size + self.horizon]

        return input_data, target

    def build_model(self, input_shape):
        """Build the MLP model.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        model : tensorflow.keras.Model
            The compiled Keras model
        """
        input_layer, output_layer = self._network.build_network(input_shape)

        # Add a dense layer for prediction
        # Output shape will be (n_samples, horizon, n_channels)
        prediction_layer = tf.keras.layers.Dense(
            self.horizon * input_shape[-1],  # horizon * n_channels
            activation="linear",
            use_bias=True,
        )(output_layer)

        # Reshape to (n_samples, horizon, n_channels)
        prediction_layer = tf.keras.layers.Reshape((self.horizon, input_shape[-1]))(
            prediction_layer
        )

        model = tf.keras.Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _fit(self, y, exog=None):
        """Fit the MLP model to the training data.

        Parameters
        ----------
        y : np.ndarray
            Training time series data
        exog : np.ndarray, default=None
            Optional exogenous variables (not supported yet)

        Returns
        -------
        self : MLPForecaster
            Reference to self
        """
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")

        # Prepare input and target data
        input_data, target = self._prepare_data(y)

        # Build the model if not already built
        if self.model_ is None:
            self.model_ = self.build_model(input_shape=input_data.shape[1:])

        # Set random seed for reproducibility
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)

        # Train the model
        self.history = self.model_.fit(
            input_data,
            target,
            batch_size=self.batch_size,
            epochs=100,  # This could be made a parameter
            validation_split=0.2,
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

        # Prepare input data using the last window_size points
        input_data = y[-self.window_size :]
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1, 1)
        else:
            input_data = input_data.reshape(1, self.window_size, -1)

        # Make predictions
        predictions = self.model_.predict(input_data)
        return predictions[0]  # Return shape: (horizon, n_channels)
