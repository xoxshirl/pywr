from pywr.parameters import Parameter, load_parameter
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ANNParameter(Parameter):
    def __init__(self, model, inputs, layer_sizes, **kwargs):
        super().__init__(model, **kwargs)

        for input in inputs:
            self.children.add(input)
        self.inputs = inputs
        self.layer_sizes = layer_sizes
        self.weights = None

    @property
    def num_layers(self):
        return len(self.layer_sizes)

    def setup(self):
        super().setup()
        outputs = [o for o in self.parents if isinstance(o, ANNOutputParameter)]

        self.weights = {
            'in': np.zeros((len(self.inputs), self.layer_sizes[0])),
            'out': np.zeros((self.layer_sizes[-1], len(outputs)))
        }

        for layer in range(self.num_layers-1):
            shp = (self.layer_sizes[layer], self.layer_sizes[layer+1])
            self.weights[layer] = np.zeros(shp)

        size = sum([w.size for w in self.weights.values()])

        self.double_size = size
        self.integer_size = 0
        self.outputs = outputs
        self.output_values = np.zeros((len(self.model.scenarios.combinations), len(outputs)))

    def value(self, ts, si):

        inputs = [i.get_value(si) for i in self.inputs]

        lyr = sigmoid(np.dot(inputs, self.weights['in']))

        for layer in range(self.num_layers-1):
            lyr = sigmoid(np.dot(lyr, self.weights[layer]))

        self.output_values[si.global_id, :] = sigmoid(np.dot(lyr, self.weights['out']))
        return 0.0
    
    def get_double_lower_bounds(self):
        return np.ones(self.double_size)*-1

    def get_double_upper_bounds(self):
        return np.ones(self.double_size)

    def set_double_variables(self, values):

        i = 0
        for key, weights in self.weights.items():
            s = weights.size
            weights[...] = values[i:i+s].reshape(weights.shape)
            i += s

    def get_double_variables(self):
        values = np.empty(self.double_size)

        i = 0
        for key, weights in self.weights.items():
            s = weights.size
            values[i:i+s] = weights.reshape(s)
            i += s

        return values

    @classmethod
    def load(cls, model, data):

        inputs = [load_parameter(model, d) for d in data.pop('inputs')]
        return cls(model, inputs=inputs, **data)
ANNParameter.register()


class ANNOutputParameter(Parameter):
    def __init__(self, model, neural_network, output_index, **kwargs):
        super().__init__(model, **kwargs)

        self.children.add(neural_network)
        self.neural_network = neural_network
        self.output_index = output_index

    def value(self, ts, si):
        return self.neural_network.output_values[si.global_id, self.output_index]

    @classmethod
    def load(cls, model, data):
        neural_network = load_parameter(model, data.pop('neural_network'))
        return cls(model, neural_network, **data)
ANNOutputParameter.register()
