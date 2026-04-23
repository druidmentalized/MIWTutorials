from enum import Enum


def batch_train(nn, X, y, learning_rate, reg_lambda, _):
    output = nn.forward(X)
    nn.backward(X, y, output, learning_rate, reg_lambda)
    return output


def mini_batch_train(nn, X, y, learning_rate, reg_lambda, batch_size):
    for i in range(0, len(X), batch_size):
        x_sample = X[i:i + batch_size]
        y_sample = y[i:i + batch_size]
        output_sample = nn.forward(x_sample)
        nn.backward(x_sample, y_sample, output_sample, learning_rate, reg_lambda)
    return nn.forward(X)


class TrainMode(Enum):
    BATCH = ("batch", batch_train)
    STOCHASTIC = ("stochastic", mini_batch_train)

    def __init__(self, label, train):
        self.label = label
        self.train = train

    def execute(self, nn, X, y, learning_rate, reg_lambda, batch_size):
        return self.train(nn, X, y, learning_rate, reg_lambda, batch_size)
