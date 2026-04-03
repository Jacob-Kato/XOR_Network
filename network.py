import neuron
class XORNetwork:
    def __init__ (self,):
        self.hidden1 = neuron.Neuron(2)
        self.hidden2 = neuron.Neuron(2)
        self.output = neuron.Neuron(2)

    
    def forward(self, x):
        a1 = self.hidden1.forward(x)
        a2 = self.hidden2.forward(x)
        y_prediction = self.output.forward([a1,a2])
        return y_prediction
    
    def backward(self,x,target,learning_rate):
        prediction = self.output.last_activation
        error = prediction - target
        delta3 = error * self.output.activation_derivative(self.output.last_z)
        delta1 = delta3 * self.output.weights[0] * self.hidden1.activation_derivative(self.hidden1.last_z)
        delta2 = delta3 * self.output.weights[1] * self.hidden2.activation_derivative(self.hidden2.last_z)
        #computing gradients for output
        gradient_ow5 = delta3 * self.hidden1.last_activation
        gradient_ow6 = delta3 * self.hidden2.last_activation
        gradient_ob = delta3 
        #computing gradients for hidden 1
        gradient_hw1 = delta1 * x[0]
        gradient_hw2 = delta1 * x[1]
        gradient_hb1 = delta1 
        #computing gradients for hidden 2
        gradient_hw3 = delta2 * x[0]
        gradient_hw4 = delta2 * x[1]
        gradient_hb2 = delta2 
        #updating output weights bias
        self.output.weights[0] -= learning_rate * gradient_ow5
        self.output.weights[1] -= learning_rate * gradient_ow6
        self.output.bias -= learning_rate * gradient_ob
        #updating hidden1 weights and bias
        self.hidden1.weights[0] -= learning_rate * gradient_hw1
        self.hidden1.weights[1] -= learning_rate * gradient_hw2
        self.hidden1.bias -= learning_rate * gradient_hb1
        #updating hidden2 weights and bias
        self.hidden2.weights[0] -= learning_rate * gradient_hw3
        self.hidden2.weights[1] -= learning_rate * gradient_hw4
        self.hidden2.bias -= learning_rate * gradient_hb2