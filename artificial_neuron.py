import math
 
def sigmoid(x):
    return 1/(1+ math.exp(-x))


def activation_function(inputs,weights):
    #net input
    sum = 0
    for input, weight in zip(inputs,weights):
        sum += input * weight

    #activation
    return sigmoid(sum)


if __name__ == "__main__":

    inputs = [0.5,0.3,0.2]
    weights = [0.4,0.7,0.2]

    output = activation_function(inputs,weights)

    print("Hello neuron:")
    print(output)



