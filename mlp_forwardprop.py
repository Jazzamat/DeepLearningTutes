import re

import numpy as np
from random import random

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



    def train(self,inputs,targets,epochs,learning_rate):

        sum_error = 0
        for i in range(epochs):
            for j, (input,target) in enumerate(zip(inputs,targets)):
              
                # perform forward prop
                output = self.forward_propagate(inputs)

                # calcualte the error
                error = target - output

                #perform back prop
                self.back_propagate(error)

                #apply gradient decent
                self.gradient_decent(learning_rate)

                #report the error
                sum_error += self._mse(target, output)

            print(f"Error: {sum_error/len(inputs)} at epoch {i}")

        return


    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


    def gradient_decent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]

         
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

        
            # self.weights[i] = weights
        


    def back_propagate(self,error, verbose=False):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error* self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0],-1).T

            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1) 

            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)

            error = np.dot(delta,self.weights[i].T)

           
            


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

    inputs = np.array([[random()/2 for _ in range(2)] for _ in range (2000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])


    #train the network
    mlp.train(inputs, targets, 5000000, 0.1)
 
    # print results
