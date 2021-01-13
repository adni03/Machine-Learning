import numpy as np


def predict(weights, inputs):

    summation = np.dot(inputs, weights[1:]) + weights[0]

    if summation > 0:

      activation = 1

    else:

      activation = 0            

    return activation

        

def train(weights,inputs, labels, learning_rate, epochs):

    for i in range(epochs):

        for j in range(len(inputs)):

            prediction = predict(weights,inputs[j])

            weights[1:] += learning_rate * (labels[j] - prediction) * inputs[j]

            weights[0] += learning_rate * (labels[j] - prediction)



inputs = []

inputs.append(np.array([1, 1]))

inputs.append(np.array([1, 0]))

inputs.append(np.array([0, 1]))

inputs.append(np.array([0, 0]))


labels = np.array([1, 0, 0, 0])

no_of_inputs=2

weights = np.zeros(no_of_inputs + 1)

learning_rate=0.2

epochs=100

train(weights,inputs, labels, learning_rate, epochs)


print("WEIGHTS : ", weights)


inputs = np.array([1, 1])

print("INPUTS : ", inputs)

res=predict(weights,inputs)

print("OUTPUT: " + str(res))


inputs = np.array([0, 1])

print("INPUTS : ", inputs)

res=predict(weights,inputs)

print("OUTPUT: " + str(res))



inputs = np.array([1, 0])

print("INPUTS : ", inputs)

res=predict(weights,inputs)

print("OUTPUT: " + str(res))



inputs = np.array([0, 0])

print("INPUTS : ", inputs)

res=predict(weights,inputs)

print("OUTPUT: " + str(res))