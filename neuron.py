import random
import math
class Neuron:
    def __init__(self,number_of_input):
        self.weights = [random.uniform(-0.5,0.5) for i in range(number_of_input)]
        self.last_z = None
        self.bias = random.uniform(-0.5,0.5)
        self.last_activation = None

    def forward (self,input):
        z = sum(x*w for x, w in zip(input, self.weights)) + self.bias
        self.last_z = z
        a = self.activation(z)
        self.last_activation = a
        return a

    def activation (self,x):
        return 1 / (1 + math.exp(-x))
        
    def activation_derivative (self,x):
        sig = 1 / (1 + math.exp(-x))
        return sig * (1 - sig)