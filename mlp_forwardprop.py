import re

import numpy as np

# save activations and derivatives

# implement backpropagation

# implement gradient decent

# implement a train method

# train with dummy datasey

# make some predictions




class MLP:

    def __init__(self, num_input=3, num_hidden=[3,5], num_output=2):

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        layers = [self.num_input] + self.num_hidden + [self.num_output]
        
        print(layers)
        
        #initiate random weights
        self.weights = []
        
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)


        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)

        self.derivatives = derivatives



    def train(self,inputs,targers,epochs,learning_rate)


    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


    def gradient_decent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]

         


            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

        
            # self.weights[i] = weights
        return


    def back_propagate(self,error, verbose=False):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error* self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0],-1).T

            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1) 

            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)

            error = np.dot(delta,self.weights[i].T)

            if verbose:
                print(f"Derivatives for W{i}:{self.derivatives[i]}")


        return error

    def _sigmoid_derivative(self,x):
        return x*(1-x)


    def forward_propagate(self,inputs):

        #first layer
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            #net input
            net_input = np.dot(activations,w)
            #activation function
            activations = self._sigmoid(net_input)
            self.activations[i+1] = activations

        return activations





    
if __name__ == "__main__":

    # create an mlp
    mlp = MLP(2, [5], 1)

    # create some inputs
    inputs = np.array([0.1,0.2])
    target = np.array([0.3])

    # perform forward prop
    output = mlp.forward_propagate(inputs)

    # calcualte the error

    error = target - output

    

    #perform back prop
    mlp.back_propagate(error, True)


    #apply gradient decent
    mlp.gradient_decent(learning_rate=1)


    #train the network


    # print results
    print(output)