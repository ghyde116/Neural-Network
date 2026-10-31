import numpy
import scipy.special 

class NeuralNetwork:

    # initialise the neural network
    def __init__ (self, inputNodes: int, hiddenNodes: int, outputNodes: int, learningRate: float):

        # set number of nodes in each layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # learning rate
        self.lr = learningRate

        # weights inside array as [w, i, j] where link is from node i to j in the next layer
        # weights sampled from a normal distribution centred around zero
        # standard deviation linked to the number of incoming node links  
        self.wih = numpy.random.normal (0.0, pow (self.iNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = numpy.random.normal (0.0, pow (self.hNodes, -0.5), (self.oNodes, self.hNodes))

        # activation function is the sigmoid function
        # create anonymous function
        self.activationFunction = lambda x: scipy.special.expit (x)

        pass

    # train the neural network
    def train ():
        pass

    # query the neural network
    def query (self, inputsList):

        # convert inputsList into 2d array
        inputs = numpy.array (inputsList, ndmin=2).T

        # calculate signals into hidden layer
        hiddenInputs = numpy.dot (self.wih, inputs)

        # calculate the signals emerging from the hidden layer
        hiddenOutputs = self.activationFunction (hiddenInputs)

        # calculate signals into final output layer
        finalInputs = numpy.dot (self.who, hiddenOutputs)

        # calculate the signals emerging from the final output layer
        finalOutputs = self.activationFunction (finalInputs)

        return finalOutputs
        

inputNodes = 3
hiddenNodes = 3
outputNodes = 3

learningRate = 0.3

n = NeuralNetwork (inputNodes, hiddenNodes, outputNodes, learningRate)