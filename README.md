# Time Series Forecasting Exploration

I have made this repository as an exploration of [Aeon](https://www.aeon-toolkit.org/)'s GSoC Project: [Project #3: Forecasting - Deep learning for forecasting](https://github.com/aeon-toolkit/aeon-admin/blob/main/gsoc/gsoc-2025-projects.md#project-3-forecasting---deep-learning-for-forecasting). It includes implementation of three deep learning based forecasting models, following the code structure of aeon's codebase.

To see how you can run this locally, please refer to the [Installation](#installation) section.

## Technical details

### Base class: `BaseDeepForecaster`

The `BaseDeepForecaster` is a base class for all the deep learning based forecasting models. It in itself is not a forecasting model, but it defines the skeleton of a forecasting model. Moreover, it inherits from `BaseForecaster`, which is a base class for all the forecasting models in aeon.

The `BaseDeepForecaster` is initialized with the following parameters:

- `horizon`: The horizon of the forecasting model.
- `batch_size`: The batch size of the forecasting model.
- `last_file_name`: The name of the last file of the forecasting model.
- `random_state`: The random state of the forecasting model.
- `axis`: The axis of the forecasting model.

I created it as an inspiration from Aeon's other base classes, such as [`BaseDeepClassifier`](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.deep_learning.base.BaseDeepClassifier.html).

All the deep learning based forecasting models inherit from `BaseDeepForecaster` class.

### General model:

All the deep learning based forecasting models inherit from `BaseDeepForecaster` class. All the models are minimal implementations of the forecasting models, with only the necessary methods to be a forecasting model.

All the models try to follow a similar structure, with the API being the same for all the models.

### Model: `MLPForecaster`

The `MLPForecaster` is a forecasting model that uses a multi-layer perceptron (MLP) to make predictions. 

### Model: `TCNForecaster`

The `TCNForecaster` is a forecasting model that uses a temporal convolutional network (TCN) to make predictions.

### Model: `LSTMForecaster`

The `LSTMForecaster` is a forecasting model based on the _(encoder-decoder)_ architecture that uses a long short-term memory (LSTM) and MLP layers to make predictions.

## Installation

1. This project depends upon Aeon, so please start by installing it.

    ```bash
    pip install -U aeon[all_extras]
    ```

    You can find the complete installation guide on [Aeon's website](https://www.aeon-toolkit.org/en/latest/installation.html).

2. Now please clone this repo locally.

    ```bash
    git clone https://github.com/inclinedadarsh/tsf-exploration.git
    ```

    Or if you have forked the repository, then you can do:

    ```bash
    git clone https://github.com/<your-username>/tsf-exploration.git
    ```

    After you're done getting a local clone, please change the directory to the project directory.

    ```bash
    cd tsf-exploration
    ```

3. Now install the project:
    ```bash
    pip install -e .
    ```

## Usage

The examples are given inside the `examples/` directory. There are three `ipython` notebook files, corresponding to each model:
1. `MLPForecaster`: `examples/mlp_forecaster.ipynb`
2. `TCNForecaster`: `examples/tcn_forecaster.ipynb`
3. `LSTMForecaster`: `examples/lstm_forecaster.ipynb`

Each of these notebooks have the code to import the data, train the model, and make predictions.

