from math import exp


class Perceptron:
    def __init__(self, input_weights, learning_rate, threshold):
        self.input_weights = input_weights
        self.learning_rate = learning_rate
        self.threshold = threshold

    def update_weights(self, update):
        self.input_weights = [
            a + b for a, b in zip(update, self.input_weights)
        ]

    def learn(self, inputs, true_value):
        """ Perceptron training rule """
        output = self.output(inputs)
        error = true_value - output
        weight_update = [i * self.learning_rate * error for i in inputs]
        self.update_weights(weight_update)

    def output(self, inputs):
        return 1 if sum(a * b for a, b in zip(
            inputs, self.input_weights)) + self.threshold > 0 else -1

    def gradient_descent(self, dataset):
        """Run gradient descent on a list of data vectors
        #arguments
            dataset: list of data vectors of the form [((data), category)*]
        """
        errors = [(i[1] - self.output(i[0])) for i in dataset]
        weight_updates = [0 for _ in self.input_weights]
        for n, elem in enumerate(errors):
            inputs = dataset[n][0]
            for i, weight in enumerate(weight_updates):
                weight_updates[i] += -elem * inputs[i] * -self.learning_rate
        self.input_weights = [
            a + b for a, b in zip(self.input_weights, weight_updates)
        ]


class UnthresholdedPerceptron(Perceptron):
    def output(self, inputs):
        return sum(a * b for a, b in zip(inputs, self.input_weights))


class SigmoidPerceptron(Perceptron):
    def output(self, inputs):
        raw = sum(a * b for a, b in zip(inputs, self.input_weights))
        return 1 / (1 + (exp(-raw)))


class Network:
    def __init__(self, layers):
        """A neural network composed of perceptrons
        #arguments
            layers: 2D list containing lists of perceptrons in each layer
                    The input layer should be the first element of the list
        """
        self.layers = layers

    def backprop(self, inputs, category):
        """Run backpropagation on a single vector
        #arguments
            inputs: list of values as long as the outermost network layer
            category: expected output value
        """
        output = self.classify(list(inputs))
        finalOut = output[-1][0]

        def sigmoid_error(output, true):
            return output * (1 - output) * (true - output)

        outputError = sigmoid_error(finalOut, category)

        def calc_errors(output, prev_error):
            return output * (1 - output) * prev_error * 0.1

        middleErrors = [
            calc_errors(output[-2][i], outputError) for i in range(2)
        ]

        errors = [middleErrors, [outputError]]
        for i, layer in enumerate(self.layers):
            for j, perceptron in enumerate(layer):
                print(list(zip(errors[i], output[i])))
                perceptron.update_weights([
                    a * errors[i][j] * perceptron.learning_rate
                    for a in output[i]
                ])

    def classify(self, vector):
        output = [vector]
        for layer in self.layers:
            output.append([i.output(output[-1]) for i in layer])
        return output


up = UnthresholdedPerceptron([0, 0], 0.5, 0)
dataset = (((1, 0), 1), ((0, 1), 1), ((0, 0), -1))
up.gradient_descent(dataset)

layer1 = [SigmoidPerceptron([0.1, 0.1], 0.5, 0) for i in range(2)]
layer2 = [SigmoidPerceptron([0.1, 0.1], 0.5, 0)]

network = Network([layer1, layer2])
