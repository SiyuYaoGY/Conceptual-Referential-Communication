import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Agent():

    def __init__(self, InterSize, NumSensors):
        self.InterSize = np.copy(InterSize)
        self.NumSensors = np.copy(NumSensors)
        self.States = np.zeros(self.InterSize+2)            # state of the neurons
        self.TimeConstants = np.ones(self.InterSize + 2)      # time-constant for each neuron
        self.invTimeConstants = 1.0/self.TimeConstants
        self.Biases = np.zeros(self.InterSize + 2)            # bias for each neuron
        self.Weights1 = np.zeros((self.NumSensors,self.InterSize))              # connection weight from sensors to interneurons
        self.Weights2 = np.zeros((self.InterSize,self.InterSize))              # connection weight across interneurons
        self.Weights3 = np.zeros((self.InterSize,2))              # connection weight from interneurons to output
        self.Outputs = np.zeros(self.InterSize + 2)           # neuron outputs
        self.Inputs = np.zeros(3) # external input to the sensor neurons

    def setWeights(self, weights1, weights2, weights3):
        self.Weights1 =  np.copy(weights1)
        self.Weights2 =  np.copy(weights2)
        self.Weights3 =  np.copy(weights3)

    def setBiases(self, biases):
        self.Biases =  np.copy(biases)

    def setTimeConstants(self, timeconstants):
        self.TimeConstants =  np.copy(timeconstants)
        self.invTimeConstants = 1.0/self.TimeConstants

    def randomizeParameters(self):
        self.Weights1 = np.random.uniform(-1,1,size=(self.NumSensors,self.InterSize))
        self.Weights2 = np.random.uniform(-1,1,size=(self.InterSize, self.InterSize))
        self.Weights3 = np.random.uniform(-1,1,size=(self.InterSize,2))
        self.Biases = np.random.uniform(-1,1,size=(self.InterSize + 2))
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.InterSize + 2))
        self.invTimeConstants = 1.0/self.TimeConstants

    def initializeState(self, s):
        self.States = np.copy(s)
        self.Outputs = sigmoid(self.States+self.Biases)
        
    def step(self, dt, inputs):
        self.Inputs = np.copy(inputs)
        inter_net_input = np.dot(self.Weights1.T, self.Inputs) + np.dot(self.Weights2.T, self.Outputs[0:self.InterSize])
        motor_net_input = np.dot(self.Weights3.T, self.Outputs[:self.InterSize])
        self.States[:self.InterSize] += dt * (self.invTimeConstants[0:self.InterSize]*(
            -self.States[:self.InterSize] + inter_net_input
            )
        )
        self.States[self.InterSize:] += dt * (self.invTimeConstants[self.InterSize:]*(
            -self.States[self.InterSize:] + motor_net_input
            )
        )
        self.Outputs = sigmoid(self.States+self.Biases)

    def save(self, filename):
        np.savez(filename, weights=self.Weights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Weights = params['weights']
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants
