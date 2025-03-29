"""Long Short-Term Memory Network (LSTMNetwork)."""

import typing

import numpy as np

from aeon.networks import BaseDeepLearningNetwork


class LSTMNetwork(BaseDeepLearningNetwork):
    """Establish the network structure for an LSTM.

    Parameters
    ----------
    n_layers : int, optional (default=3)
        The number of LSTM layers in the network.
    n_units : Union[int, List[int]], optional (default=100)
        Number of units in each LSTM layer.
    activation : Union[str, List[str]], optional (default='tanh')
        Activation function(s) for each LSTM layer.
    recurrent_activation : Union[str, List[str]], optional (default='sigmoid')
        Recurrent activation function(s) for each LSTM layer.
    dropout_rate : Union[float, List[Union[int, float]]], optional (default=None)
        Dropout rate(s) for each LSTM layer. If None, a default rate of 0.2 is used,
        except the first element, being 0.1. Dropout rate(s) are typically a number
        in the interval [0, 1].
    dropout_last : float, default = 0.3
        The dropout rate of the last layer.
    return_sequences : bool, default = True
        Whether to return the last output in the output sequence, or the full sequence.
    use_bias : bool, default = True
        Condition on whether or not to use bias values for LSTM layers.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        n_layers: int = 3,
        n_units: typing.Union[int, list[int]] = 100,
        activation: typing.Union[str, list[str]] = "tanh",
        recurrent_activation: typing.Union[str, list[str]] = "sigmoid",
        dropout_rate: typing.Union[float, list[float]] = None,
        dropout_last: float = None,
        return_sequences: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.dropout_rate = dropout_rate
        self.dropout_last = dropout_last
        self.return_sequences = return_sequences
        self.use_bias = use_bias

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple of shape = (n_timepoints (m), n_channels (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        if isinstance(self.activation, str):
            self._activation = [self.activation] * self.n_layers
        elif isinstance(self.activation, list):
            assert len(self.activation) == self.n_layers, (
                "There should be an `activation` function associated with each layer."
            )
            assert all(isinstance(a, str) for a in self.activation), (
                "Activation must be a list of strings."
            )
            self._activation = self.activation

        if isinstance(self.recurrent_activation, str):
            self._recurrent_activation = [self.recurrent_activation] * self.n_layers
        elif isinstance(self.recurrent_activation, list):
            assert len(self.recurrent_activation) == self.n_layers, (
                "There should be a `recurrent_activation` function associated with each layer."
            )
            assert all(isinstance(a, str) for a in self.recurrent_activation), (
                "Recurrent activation must be a list of strings."
            )
            self._recurrent_activation = self.recurrent_activation

        if self.dropout_rate is None:
            self._dropout_rate = [0.1]
            self._dropout_rate.extend([0.2] * (self.n_layers - 1))
            assert np.all(np.array(self._dropout_rate) - 1 <= 0), (
                "Dropout rate(s) should be in the interval [0, 1]."
            )
        elif isinstance(self.dropout_rate, (int, float)):
            self._dropout_rate = [float(self.dropout_rate)] * self.n_layers
            assert np.all(np.array(self._dropout_rate) - 1 <= 0), (
                "Dropout rate(s) should be in the interval [0, 1]."
            )
        elif isinstance(self.dropout_rate, list):
            assert len(self.dropout_rate) == self.n_layers, (
                "There should be a `dropout_rate` associated with each layer."
            )
            assert all(isinstance(d, (int, float)) for d in self.dropout_rate), (
                "Dropout rates must be int or float."
            )
            assert len(self.dropout_rate) == self.n_layers, (
                "Dropout list length must match number of layers."
            )
            self._dropout_rate = [float(d) for d in self.dropout_rate]
            assert np.all(np.array(self._dropout_rate) - 1 <= 0), (
                "Dropout rate(s) should be in the interval [0, 1]."
            )

        if isinstance(self.n_units, int):
            self._n_units = [self.n_units] * self.n_layers
        elif isinstance(self.n_units, list):
            assert all(isinstance(u, int) for u in self.n_units), (
                "`n_units` must be int for all layers."
            )
            assert len(self.n_units) == self.n_layers, (
                "`n_units` length must match number of layers."
            )
            self._n_units = self.n_units

        if self.dropout_last is None:
            self._dropout_last = 0.3
        else:
            assert isinstance(self.dropout_last, float) or (
                int(self.dropout_last // 1) in [0, 1]
            ), "a float is expected in the `dropout_last` argument."
            assert self.dropout_last - 1 <= 0, (
                "`dropout_last` argument must be a number in the interval [0, 1]"
            )
            self._dropout_last = self.dropout_last

        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)
        x = input_layer

        for idx in range(0, self.n_layers):
            # For intermediate layers, always return sequences
            # For the last layer, return sequences based on the parameter
            return_sequences = (
                True if idx < self.n_layers - 1 else self.return_sequences
            )

            x = keras.layers.LSTM(
                self._n_units[idx],
                activation=self._activation[idx],
                recurrent_activation=self._recurrent_activation[idx],
                return_sequences=return_sequences,
                use_bias=self.use_bias,
            )(x)

            # Apply dropout only to intermediate layers
            # if idx < self.n_layers - 1:
            #     x = keras.layers.Dropout(self._dropout_rate[idx])(x)

        # Apply final dropout to the output
        # output_layer = keras.layers.Dropout(self._dropout_last)(x)
        # FIXME: DROPOUT IS NOT WORKING
        output_layer = x

        return input_layer, output_layer
