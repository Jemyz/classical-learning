import numpy as np


class MLP():
    def __init__(self, layers, weights_init):
        self.weights = []
        self.biases = []

        # initialize weights
        self.num_layers = len(layers) - 1
        for l in range(self.num_layers):
            self.weights.append(np.random.uniform(weights_init[0], weights_init[1], [layers[l], layers[l + 1]]))
            self.biases.append(np.zeros(layers[l + 1]))
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    def train(self, X, y, epochs, print_loss):
        data_len = len(X)
        for epoch in range(epochs):
            input_i = np.array(X)
            output = [input_i]
            for i in range(self.num_layers):
                weight_i = self.weights[i]
                bias_i = self.biases[i]
                output_i = np.tanh(input_i.dot(weight_i) + bias_i)
                output.append(output_i)
                input_i = output_i

            exp_output = np.exp(output_i)
            predictions_probs = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            weight_delta = self.back_propagation(predictions_probs, output, y)
            self.weights += weight_delta
            if (print_loss and i % 1000 == 0):
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(predictions_probs, data_len), y))

    def back_propagation(self, predictions_probs, output, y, learning_rate=0.01):
        samples_len = len(y)
        deltas = []
        one_hot = np.squeeze(np.eye(np.max(y) + 1)[y])
        error = (one_hot - predictions_probs)

        output_derv = [1 - (layer * layer) for layer in output]
        deltas.append(output_derv[-1] * error)
        current_delta = deltas[-1]
        for i in reversed(range(self.num_layers)):
            new_delta = (current_delta.dot(np.transpose(self.weights[i]))) * output_derv[i]
            deltas.append(new_delta)
            current_delta = new_delta

        deltas = deltas[::-1]
        deltas = deltas[1:]
        output = output[:-1]

        weight_delta = [learning_rate * np.transpose(output[i]).dot(deltas[i] / samples_len) for i in
                        range(len(deltas))]
        return weight_delta

    def calcuate_loss(self, predictions_probs, data_len, y):
        # Calculating the loss
        corect_logprobs = -np.log(predictions_probs[range(data_len), y])
        data_loss = np.sum(corect_logprobs)
        # # Add regulatization term to loss (optional)
        # data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / data_len * data_loss

    def predict(self, X):

        input_i = np.array(X)
        for i in range(self.num_layers):
            weight_i = self.weights[i]
            bias_i = self.biases[i]
            output_i = np.tanh(input_i.dot(weight_i) + bias_i)
            input_i = output_i
        exp_output = np.exp(output_i)
        predictions_probs = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        predictions = np.argmax(predictions_probs, axis=1)
        return predictions


def read_data(directory):
    import pandas as pd

    df = pd.read_csv(directory,skiprows=1)
    X = []
    Y = []
    for row in df.values:
        x,y = row[0].split("  ")
        X.append(list(map(float,x.split(" "))))
        Y.append(list(map(float,y.split(" "))))

    return X,Y



def main():
    trainingset_dir = "PA-B-train-01.dat"
    X,Y = read_data(trainingset_dir)
    input_size = len(X[0])
    output_size = len(Y[0])
    layers_size = [input_size, 3, output_size]
    model = MLP(layers_size, [-0.5, 0.5])
    # x = [[1, 1], [1, 2], [1, 3], [1, 4]]
    # y = [1, 0, 1, 0]
    epochs = 100
    print_loss = True
    model.train(X, Y, epochs, print_loss)
    print(model.predict([[1, 2]]))




if __name__ == "__main__":
    main()
