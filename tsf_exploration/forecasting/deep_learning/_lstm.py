"""LSTM-based time series forecaster using encoder-decoder pattern."""

__all__ = ["LSTMForecaster"]

import numpy as np
import tensorflow as tf

from tsf_exploration.forecasting.deep_learning.base import BaseDeepForecaster
from tsf_exploration.networks._lstm import LSTMNetwork
from aeon.networks import MLPNetwork


class LSTMForecaster(BaseDeepForecaster):
    """Long Short-Term Memory (LSTM) based time series forecaster using encoder-decoder pattern.

    The forecaster uses an LSTM network as an encoder to process the input sequence,
    followed by an MLP decoder to generate predictions.

    Parameters
    ----------
    horizon : int, default=1
        The number of time steps ahead to forecast
    window_size : int, default=10
        The size of the sliding window used for training
    # Encoder parameters (LSTM)
    encoder_layers : int, default=2
        The number of LSTM layers in the encoder network
    encoder_units : int or list of int, default=64
        Number of units in each encoder LSTM layer
    encoder_activation : str or list of str, default='relu'
        Activation function(s) for each encoder LSTM layer
    encoder_recurrent_activation : str or list of str, default='sigmoid'
        Recurrent activation function(s) for each encoder LSTM layer
    encoder_dropout_rate : float or list of float, default=None
        Dropout rate(s) for each encoder LSTM layer
    # Decoder parameters (MLP)
    decoder_layers : int, default=2
        The number of dense layers in the decoder network
    decoder_units : int or list of int, default=64
        Number of units in each decoder dense layer
    decoder_activation : str or list of str, default='relu'
        Activation function(s) for each decoder dense layer
    decoder_dropout_rate : float or list of float, default=None
        Dropout rate(s) for each decoder dense layer
    dropout_last : float, default=0.3
        The dropout rate of the last layer
    use_bias : bool, default=True
        Whether to use bias values for layers
    batch_size : int, default=32
        Training batch size
    random_state : int, default=None
        Random seed for reproducibility
    epochs : int, default=100
        Number of epochs to train the model
    verbose : int, default=1
        Verbosity level during training
    axis : int, default=0
        The axis along which to forecast
    """

    def __init__(
        self,
        horizon=1,
        window_size=10,
        # Encoder parameters
        encoder_layers=2,
        encoder_units=64,
        encoder_activation="relu",
        encoder_recurrent_activation="sigmoid",
        encoder_dropout_rate=None,
        # Decoder parameters
        decoder_layers=2,
        decoder_units=64,
        decoder_activation="relu",
        decoder_dropout_rate=None,
        dropout_last=0.3,
        use_bias=True,
        batch_size=32,
        random_state=None,
        epochs=100,
        verbose=1,
        axis=0,
    ):
        super().__init__(
            horizon=horizon,
            batch_size=batch_size,
            random_state=random_state,
            axis=axis,
        )

        self.window_size = window_size

        # Encoder parameters
        self.encoder_layers = encoder_layers
        self.encoder_units = encoder_units
        self.encoder_activation = encoder_activation
        self.encoder_recurrent_activation = encoder_recurrent_activation
        self.encoder_dropout_rate = encoder_dropout_rate

        # Decoder parameters
        self.decoder_layers = decoder_layers
        self.decoder_units = decoder_units
        self.decoder_activation = decoder_activation
        self.decoder_dropout_rate = decoder_dropout_rate

        self.dropout_last = dropout_last
        self.use_bias = use_bias
        self.epochs = epochs
        self.verbose = verbose

        # Initialize the encoder network (LSTM)
        self._encoder = LSTMNetwork(
            n_layers=encoder_layers,
            n_units=encoder_units,
            activation=encoder_activation,
            recurrent_activation=encoder_recurrent_activation,
            dropout_rate=encoder_dropout_rate,
            dropout_last=dropout_last,
            return_sequences=False,  # Only return the last output
            use_bias=use_bias,
        )

        # Initialize the decoder network (MLP)
        self._decoder = MLPNetwork(
            n_layers=decoder_layers,
            n_units=decoder_units,
            activation=decoder_activation,
            dropout_rate=decoder_dropout_rate,
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
        """Build the model with LSTM encoder and MLP decoder pattern.

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
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Encoder (LSTM)
        x = inputs
        for i in range(self.encoder_layers):
            x = tf.keras.layers.LSTM(
                self.encoder_units
                if isinstance(self.encoder_units, int)
                else self.encoder_units[i],
                activation=self.encoder_activation
                if isinstance(self.encoder_activation, str)
                else self.encoder_activation[i],
                recurrent_activation=self.encoder_recurrent_activation
                if isinstance(self.encoder_recurrent_activation, str)
                else self.encoder_recurrent_activation[i],
                return_sequences=False if i == self.encoder_layers - 1 else True,
                use_bias=self.use_bias,
            )(x)
            if self.encoder_dropout_rate is not None:
                x = tf.keras.layers.Dropout(
                    self.encoder_dropout_rate
                    if isinstance(self.encoder_dropout_rate, float)
                    else self.encoder_dropout_rate[i]
                )(x)

        # Repeat the encoder output for each time step in the horizon
        x = tf.keras.layers.RepeatVector(self.horizon)(x)

        # Decoder (MLP)
        # First flatten the repeated vector
        x = tf.keras.layers.Flatten()(x)

        # Add dense layers
        for i in range(self.decoder_layers):
            x = tf.keras.layers.Dense(
                self.decoder_units
                if isinstance(self.decoder_units, int)
                else self.decoder_units[i],
                activation=self.decoder_activation
                if isinstance(self.decoder_activation, str)
                else self.decoder_activation[i],
                use_bias=self.use_bias,
            )(x)
            if self.decoder_dropout_rate is not None:
                x = tf.keras.layers.Dropout(
                    self.decoder_dropout_rate
                    if isinstance(self.decoder_dropout_rate, float)
                    else self.decoder_dropout_rate[i]
                )(x)

        # Final dropout
        x = tf.keras.layers.Dropout(self.dropout_last)(x)

        # Calculate the number of features per time step
        n_features = input_shape[-1]  # Number of channels
        n_units = (
            self.decoder_units
            if isinstance(self.decoder_units, int)
            else self.decoder_units[-1]
        )

        # Reshape to (horizon, n_features)
        x = tf.keras.layers.Dense(self.horizon * n_features)(x)
        x = tf.keras.layers.Reshape((self.horizon, n_features))(x)

        # Create and compile the model
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _fit(self, y, exog=None):
        """Fit the LSTM model to the training data.

        Parameters
        ----------
        y : np.ndarray
            Training time series data
        exog : np.ndarray, default=None
            Optional exogenous variables (not supported yet)

        Returns
        -------
        self : LSTMForecaster
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
            verbose=self.verbose,
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
            Forecasted values with shape (horizon, n_channels)
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
