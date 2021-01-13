import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def fwd_prop(ip, weights, bias):
    weighted_sum = np.dot(ip, weights) + bias
    return weighted_sum

def threshold(num):
    if(num > 0):
        return 1
    elif(num < 0):
        return -1

def backprop(weights, bias, ip, real, guess, a):
    for i in range(len(weights)):
        weights[i] = weights[i] + (real - guess)*ip[i]*a
    bias = bias + (real - guess)*a
    return bias


def train(ip, op, weights, bias, a):
    for i in range(len(ip)):
        ws = fwd_prop(ip[i], weights, bias)
        clas = threshold(ws)
        bias = backprop(weights, bias, ip[i], op[i], clas, a)


df = pd.read_csv('percep_data.csv')
# df2 = pd.DataFrame()
clas = df['Y']
clas = clas.to_numpy()
df = df.drop(['Y'], axis=1)
df = df.to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(df, clas, test_size=0.1)

weights = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
bias = random.uniform(-1, 1)
alpha = 0.5

print(weights)

for z in range(25):
    train(df, clas, weights, bias, alpha)

print(weights)

predn = []
for i in range(len(xtest)):
    ws = fwd_prop(xtest[i], weights, bias)
    clas = threshold(ws)
    predn.append(clas)

predn = np.array(predn)

accuracy = accuracy_score(ytest, predn)
print(accuracy)