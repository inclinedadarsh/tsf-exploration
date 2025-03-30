"""TCN-based time series forecaster."""

__all__ = ["TCNForecaster"]

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tcn import TCN

from tsf_exploration.forecasting.deep_learning.base import BaseDeepForecaster


class TCNForecaster(BaseDeepForecaster):
    """Temporal Convolutional Network (TCN) based time series forecaster.

    Parameters
    ----------
    horizon : int, default=1
        The number of time steps ahead to forecast
    window_size : int, default=10
        The size of the sliding window used for training
    nb_filters : int, default=64
        Number of filters in the TCN layer
    kernel_size : int, default=3
        Size of the convolutional kernel
    nb_stacks : int, default=1
        Number of TCN stacks
    dropout_rate : float, default=0.0
        Dropout rate for the TCN layer
    activation : str, default='relu'
        Activation function for the TCN layer
    use_batch_norm : bool, default=False
        Whether to use batch normalization
    use_layer_norm : bool, default=False
        Whether to use layer normalization
    use_skip_connections : bool, default=True
        Whether to use skip connections
    batch_size : int, default=32
        Training batch size
    epochs : int, default=100
        Number of training epochs
    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        horizon=1,
        window_size=10,
        nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dropout_rate=0.0,
        activation="relu",
        use_batch_norm=False,
        use_layer_norm=False,
        use_skip_connections=True,
        batch_size=32,
        epochs=100,
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
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections
        self.epochs = epochs

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
        """Build the TCN model.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        model : tensorflow.keras.Model
            The compiled Keras model
        """
        # Input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)

        # TCN layer
        tcn_layer = TCN(
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=self.nb_stacks,
            padding="causal",
            use_skip_connections=self.use_skip_connections,
            dropout_rate=self.dropout_rate,
            return_sequences=False,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
        )(input_layer)

        # Dense layer for prediction
        # Output shape will be (n_samples, horizon * n_channels)
        prediction_layer = Dense(
            self.horizon * input_shape[-1],  # horizon * n_channels
            activation="linear",
            use_bias=True,
        )(tcn_layer)

        # Reshape to (n_samples, horizon, n_channels)
        prediction_layer = Reshape((self.horizon, input_shape[-1]))(prediction_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _fit(self, y, exog=None):
        """Fit the TCN model to the training data.

        Parameters
        ----------
        y : np.ndarray
            Training time series data
        exog : np.ndarray, default=None
            Optional exogenous variables (not supported yet)

        Returns
        -------
        self : TCNForecaster
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
            epochs=self.epochs,
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
