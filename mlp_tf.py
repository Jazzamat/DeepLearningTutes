from re import L
import numpy as np
from random import random
from sklearn.model_selection import train_test_split 
import tensorflow as tf


# TODO compile model
# TODO train model
# TODO evaluate model
# TODO make prediction


def generate_dataset(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range (num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = generate_dataset(5000,0.3)
    print(f"x_test: {x_test}")
    print(f"y_test: {y_test}")

    # build model 2 -> 5 -> 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")
    
    # train model
    model.fit(x_train, y_train, epochs=100)


    # evaluate model
    print("\nModel evaluation")
    model.evaluate(x_test, y_test, verbose=1)

    data = np.array([[0.1,0.2], [0.2,0.2]]) 
    predictions = model.predict(data)

    print("\nSome Predictions")
    for d, p in zip(data, predictions):
        print(f"{d[0]} + {d[1]} = {p[0]}")