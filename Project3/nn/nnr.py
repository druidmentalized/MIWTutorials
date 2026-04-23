import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from Project3.nn.enums import Activation, WeightInit, UpdateMode

# Activation functions
def relu(x): return np.maximum(0, 1 * x)
def relu_derivative(x): return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1.0 - np.tanh(x) ** 2


class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size,
                 activation=Activation.TANH,
                 weight_init=WeightInit.STANDARD,
                 update_mode=UpdateMode.BATCH,
                 random_state=67):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.update_mode = update_mode
        self.random_state = random_state

        self.weights_input_hidden, self.weights_hidden_output = self._init_weights()

        # Bias initialization for hidden and output layers
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

        # History storage for evaluation metrics
        self.history_mse = []
        self.history_r2 = []

    def _init_weights(self):
        rgen = np.random.default_rng(self.random_state)

        weights_input_hidden = rgen.standard_normal(size=(self.input_size, self.hidden_size))
        weights_hidden_output = rgen.standard_normal(size=(self.hidden_size, self.output_size))

        if self.update_mode == WeightInit.HE:
            weights_input_hidden *= np.sqrt(2.0 / self.input_size)
            weights_hidden_output *= np.sqrt(2.0 / self.hidden_size)

        return weights_input_hidden, weights_hidden_output


    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = tanh(hidden_input) if self.activation == Activation.TANH else relu(hidden_input)

        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        return tanh(output_input) if self.activation == Activation.TANH else relu(output_input)

    def backward(self, X, y, output, learning_rate, reg_lambda):
        output_error = y - output

        hidden_output = relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)

        gradient_hidden_output = np.dot(hidden_output.T, output_error)

        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error *= relu_derivative(hidden_output)
        # hidden_error *= self.tanh_derivative(hidden_output)

        gradient_input_hidden = np.dot(X.T, hidden_error)

        # Weight and bias updates with L2 regularization
        self.weights_hidden_output += (gradient_hidden_output - reg_lambda * self.weights_hidden_output) * learning_rate
        self.weights_input_hidden += (gradient_input_hidden - reg_lambda * self.weights_input_hidden) * learning_rate

        self.bias_output += np.sum(output_error, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_error, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate, reg_lambda):
        print("Start training neural networks")

        for epoch in range(1, epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate, reg_lambda)

            mse = mean_squared_error(y, output)
            r2 = r2_score(y, output)

            self.history_mse.append(mse)
            self.history_r2.append(r2)

            if epoch % (epochs // 10) == 0:
                print(f"[%] {epoch / epochs}")

            if r2 >= 0.95:
                print("Training stopped, R^2 score reached 0.95")
                break

