import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from Project3.nn.activation import Activation
from Project3.nn.train_mode import TrainMode
from Project3.nn.weight_init import WeightInit


class NeuralNetworkRegression:
    def __init__(
        self,
        input_size=1,
        hidden_size=300,
        output_size=1,
        activation=Activation.TANH,
        weight_init=WeightInit.STANDARD,
        train_mode=TrainMode.BATCH,
        random_state=67
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.train_mode = train_mode
        self.random_state = random_state

        self.rgen = np.random.default_rng(self.random_state)
        self.weights_input_hidden, self.weights_hidden_output = self._init_weights()

        # Bias initialization for hidden and output layers
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

        # History storage for evaluation metrics
        self.history_mse = []
        self.history_r2 = []

    def _init_weights(self):
        weights_input_hidden = self.weight_init.initialize(self.rgen, self.input_size, self.hidden_size)
        weights_hidden_output = self.weight_init.initialize(self.rgen, self.hidden_size, self.output_size)
        return weights_input_hidden, weights_hidden_output

    def _output(self, X, weight_in, bias):
        return self.activation.forward(np.dot(X, weight_in) + bias)

    def forward(self, X):
        hidden_output = self._output(X, self.weights_input_hidden, self.bias_hidden)
        return self._output(hidden_output, self.weights_hidden_output, self.bias_output)

    def backward(self, X, y, output, learning_rate, reg_lambda):
        output_error = y - output
        hidden_output = self._output(X, self.weights_input_hidden, self.bias_hidden)
        gradient_hidden_output = np.dot(hidden_output.T, output_error)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error *= self.activation.derivative(hidden_output)
        gradient_input_hidden = np.dot(X.T, hidden_error)

        self.weights_hidden_output += (gradient_hidden_output - reg_lambda * self.weights_hidden_output) * learning_rate
        self.weights_input_hidden += (gradient_input_hidden - reg_lambda * self.weights_input_hidden) * learning_rate

        self.bias_output += np.sum(output_error, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_error, axis=0) * learning_rate

    def _execute_training(self, X, y, learning_rate, reg_lambda, batch_size):
        return self.train_mode.execute(self, X, y, learning_rate, reg_lambda, batch_size)

    def _update_stats(self, y, output):
        r2 = r2_score(y, output)
        mse = mean_squared_error(y, output)

        self.history_r2.append(r2)
        self.history_mse.append(mse)

        return r2, mse

    def train(self, X, y, epochs=2000, learning_rate=0.005, reg_lambda=0.01, batch_size=1, r2_max=0.95):
        for epoch in range(1, epochs):
            output = self._execute_training(X, y, learning_rate, reg_lambda, batch_size)

            r2, _ = self._update_stats(y, output)

            if epoch % (epochs // 10) == 0:
                print(f"{(epoch / epochs) * 100}%")

            if 0 < r2_max <= 1 and r2 >= r2_max:
                print(f"Training stopped, R^2 score reached {r2_max}")
                break
