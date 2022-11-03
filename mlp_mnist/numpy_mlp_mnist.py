#%%
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def split_dataset(mnist, ratio_test):
    # using sklearn's train_test_split

    # split data

    # if split mnist dataset to train data & test data (9 : 1)
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=ratio_test, shuffle=True, random_state=1)

    # train data
    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # test data
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        x[self.mask] = 0
        dx = x

        return dx



class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.X = x
        out = np.matmul(x, self.W)

        return out

    def backward(self, Backpropagation_in):
        Backpropagation = np.matmul(Backpropagation_in, self.W.T)
        self.dW = np.matmul(self.X.T,Backpropagation_in)
        self.db = np.sum(Backpropagation_in, axis = 0)

        return Backpropagation



class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # soft max output
        self.t = None # one hot encoded correct label

    def cross_entropy(self, eps = 1e-7):
        return -np.sum(self.t*np.log(self.y+eps))


    def softmax(self, x):
        for i in range(x.shape[0]):
            c = np.max(x[i])
            exp_x = np.exp(x[i] -c)
            sum_exp_x = np.sum(exp_x)
            y = exp_x / sum_exp_x
            x[i] = y
        return x

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy()

        return self.y ,self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        Backpropagation = (self.y - self.t) / batch_size

        return Backpropagation


#============================================================================


def SLP(X_train, y_train, X_test, y_test, n_class_in, n_epoch_in, batch_size_in, learning_rate_in):
    # initialization params
    # input layer -- output layer
    n_class = n_class_in
    n_epoch = n_epoch_in
    batch_size = batch_size_in
    n_batch = int(X_train.shape[0]/batch_size)
    learning_rate = learning_rate_in
    Loss = []
    Accuracy = []
    Loss_test = []
    Accuracy_test = []
    parameters = {}

    #input layer -- outputlayer
    parameters['W'] = np.random.uniform(low = -1.0, high = 1.0, size=(X_train.shape[1],n_class))
    parameters['b'] = np.random.uniform(low = -1.0, high = 1.0, size = n_class)

    #Affine
    input_to_output = Affine(parameters['W'], parameters['b'])

    # Activation and loss (softmax and cross_entropy[loss] in last output layer)
    Softmax_with_loss = SoftmaxWithLoss()

    print("##########Train start!!##########")

    for epoch in range(n_epoch):
        batch_accuracy = []
        batch_loss = []
        for batch in range(n_batch):
            start_index = batch*batch_size
            end_index = batch*batch_size+batch_size
            
            # input
            X = X_train[start_index : end_index]
            Y = y_train[start_index : end_index]

            # one-hot-encoding for Y data
            Y_hot = np.array(Y, dtype = int)
            Y_hot = np.eye(10)[Y_hot]
            Y_hot = np.array(Y_hot, dtype = float)

            #feed forward
            # input layer -- output layer
            Z = input_to_output.forward(X)
            A, loss = Softmax_with_loss.forward(Z, Y_hot)

            #predict
            predict = np.argmax(A, axis = 1)
            Y = np.array(Y, dtype = int)
            accuracy = (predict == Y).mean()
            batch_accuracy.append(accuracy)

            #loss
            batch_loss.append(loss)

            #Backpropagation
            Backpropagation = Softmax_with_loss.backward()
            Backpropagation = input_to_output.backward(Backpropagation)

            #update params
            input_to_output.W = input_to_output.W - learning_rate * input_to_output.dW
            input_to_output.b = input_to_output.b - learning_rate * input_to_output.db

        # print current state
        batch_accuracy = np.array(batch_accuracy)
        mean_batch_accuracy = np.mean(batch_accuracy)
        batch_loss = np.array(batch_loss) 
        mean_batch_loss = np.mean(batch_loss)
        print("Epoch : ", epoch+1, " Train_Accuracy : ", mean_batch_accuracy, " Train_Loss : ", mean_batch_loss)
        Accuracy.append(mean_batch_accuracy)
        Loss.append(float(mean_batch_loss))


        #valid
        # one-hot-encoding for Y data
        Y_hot_test = np.array(y_test, dtype = int)
        Y_hot_test = np.eye(10)[Y_hot_test]
        Y_hot_test = np.array(Y_hot_test, dtype = float)

        #feed forward
        # input layer -- output layer
        Z_test = input_to_output.forward(X_test)
        A_test, loss_test = Softmax_with_loss.forward(Z_test, Y_hot_test)

        #predict
        predict_test = np.argmax(A_test, axis = 1)
        y_test = np.array(y_test, dtype = int)
        accuracy_test = (predict_test == y_test).mean()
        Accuracy_test.append(accuracy_test)

        #loss
        Loss_test.append(loss_test)

        print("Epoch : ", epoch+1, "Test_Accuracy : ", accuracy_test, "Test_Loss : ", loss_test)
        print("\n")

    #visualizing

    # Accuracy
    plt.plot(Accuracy, color = 'm')
    plt.xlabel('Accuracy_train')
    plt.ylabel('Epoch')
    plt.show()

    # Loss
    plt.plot(Loss, color = 'm')
    plt.xlabel('Loss_train')
    plt.ylabel('Epoch')
    plt.show()

    # Accuracy_test
    plt.plot(Accuracy_test, color = 'm')
    plt.xlabel('Accuracy_test')
    plt.ylabel('Epoch')
    plt.show()

    # Loss_test
    plt.plot(Loss_test, color = 'm')
    plt.xlabel('Loss_test')
    plt.ylabel('Epoch')
    plt.show()

    # visualizing Mnist data & label

    # one-hot-encoding for Y data
    Y_hot_test = np.array(y_test, dtype = int)
    Y_hot_test = np.eye(10)[Y_hot_test]
    Y_hot_test = np.array(Y_hot_test, dtype = float)

    #feed forward
    # input layer -- output layer
    Z_test = input_to_output.forward(X_test)
    A_test, loss_test = Softmax_with_loss.forward(Z_test, Y_hot_test)

    #predict
    predict_test = np.argmax(A_test, axis = 1)
    y_test = np.array(y_test, dtype = int)
    accuracy_test = (predict_test == y_test).mean()
    Accuracy_test.append(accuracy_test)

    for i in range(10):
        random_index = np.random.randint(0,X_test.shape[0])
        plt.imshow(X_test[random_index].reshape((28,28)), cmap='gray')
        plt.show()

        print("Predict[",i+1,"] : ", predict_test[random_index])
        print("Correct[",i+1,"] : ", y_test[random_index])
        print("\n")



def MLP(X_train, y_train, X_test, y_test, n_class_in, n_epoch_in, batch_size_in, learning_rate_in):
    # initialization params
    # input layer -- hidden layer1  -- hidden layer2 -- output layer
    n_class = n_class_in
    hidden_layer = [500,500]
    n_epoch = n_epoch_in
    batch_size= batch_size_in
    n_batch = int(X_train.shape[0]/batch_size)
    learning_rate = learning_rate_in
    Loss = []
    Accuracy = []
    Loss_test = []
    Accuracy_test = []
    parameters = {}

    # input layer -- hidden layer1
    parameters['W1'] = np.random.uniform(low = -1.0, high = 1.0, size=(X_train.shape[1], hidden_layer[0]))
    parameters['b1'] = np.random.uniform(low = -1.0, high = 1.0, size = hidden_layer[0])

    # hidden layer1 -- hidden layer2
    parameters['W2'] = np.random.uniform(low = -1.0, high = 1.0, size = (hidden_layer[0], hidden_layer[1]))
    parameters['b2'] = np.random.uniform(low = -1.0, high = 1.0, size = (hidden_layer[1]))

    # hidden layer2 -- output layer
    parameters['W3'] = np.random.uniform(low = -1.0, high = 1.0, size = (hidden_layer[1], n_class))
    parameters['b3'] = np.random.uniform(low = -1.0, high = 1.0, size = n_class)

    # Affine
    input_to_hidden1 = Affine(parameters['W1'], parameters['b1'])
    hidden1_to_hidden2 = Affine(parameters['W2'], parameters['b2'])
    hidden2_to_output = Affine(parameters['W3'], parameters['b3'])

    # Activation
    Activation_func = Relu()

    # Activation and loss (softmax and cross_entropy[loss] in last output layer)
    Softmax_with_loss = SoftmaxWithLoss()

    

    print("##########Train start!!##########")

    for epoch in range(n_epoch):
        batch_accuracy = []
        batch_loss = []
        for batch in range(n_batch):
            start_index = batch*batch_size
            end_index = batch*batch_size+batch_size
            
            # input
            X = X_train[start_index : end_index]
            Y = y_train[start_index : end_index]

            # one-hot-encoding for Y data
            Y_hot = np.array(Y, dtype = int)
            Y_hot = np.eye(10)[Y_hot]
            Y_hot = np.array(Y_hot, dtype = float)

            #feed forward

            # input layer -- hidden layer1 
            Z1 = input_to_hidden1.forward(X)
            A1 = Activation_func.forward(Z1)

            # hidden layer1 -- hidden layer2
            Z2 = hidden1_to_hidden2.forward(A1)
            A2 = Activation_func.forward(Z2)

            # hidden layer2 -- output layer
            Z3 = hidden2_to_output.forward(A2)
            A3, loss = Softmax_with_loss.forward(Z3, Y_hot)


            #predict
            predict = np.argmax(A3, axis = 1)
            Y = np.array(Y, dtype = int)
            accuracy = (predict == Y).mean()
            batch_accuracy.append(accuracy)

            
            #loss
            batch_loss.append(loss)


            #Backpropagation
            Backpropagation = Softmax_with_loss.backward()
            Backpropagation = hidden2_to_output.backward(Backpropagation)

            Backpropagation = Activation_func.backward(Backpropagation)
            Backpropagation = hidden1_to_hidden2.backward(Backpropagation)

            Backpropagation = Activation_func.backward(Backpropagation)
            Backpropagation = input_to_hidden1.backward(Backpropagation)


            #update params
            hidden2_to_output.W = hidden2_to_output.W - learning_rate * hidden2_to_output.dW
            hidden2_to_output.b = hidden2_to_output.b - learning_rate * hidden2_to_output.db

            hidden1_to_hidden2.W = hidden1_to_hidden2.W - learning_rate * hidden1_to_hidden2.dW
            hidden1_to_hidden2.b = hidden1_to_hidden2.b - learning_rate * hidden1_to_hidden2.db

            input_to_hidden1.W = input_to_hidden1.W - learning_rate * input_to_hidden1.dW
            input_to_hidden1.b = input_to_hidden1.b - learning_rate * input_to_hidden1.db


        # print current state
        batch_accuracy = np.array(batch_accuracy)
        mean_batch_accuracy = np.mean(batch_accuracy)
        batch_loss = np.array(batch_loss) 
        mean_batch_loss = np.mean(batch_loss)
        print("Epoch : ", epoch+1, " Train_Accuracy : ", mean_batch_accuracy, " Train_Loss : ", mean_batch_loss)
        Accuracy.append(mean_batch_accuracy)
        Loss.append(float(mean_batch_loss))

        #valid
        # one-hot-encoding for Y data
        Y_hot_test = np.array(y_test, dtype = int)
        Y_hot_test = np.eye(10)[Y_hot_test]
        Y_hot_test = np.array(Y_hot_test, dtype = float)

        #feed forward

        # input layer -- hidden layer1 
        Z1_test = input_to_hidden1.forward(X_test)
        A1_test = Activation_func.forward(Z1_test)

        # hidden layer1 -- hidden layer2
        Z2_test = hidden1_to_hidden2.forward(A1_test)
        A2_test = Activation_func.forward(Z2_test)

        # hidden layer2 -- output layer
        Z3_test = hidden2_to_output.forward(A2_test)
        A3_test, loss_test = Softmax_with_loss.forward(Z3_test, Y_hot_test)


        #predict
        predict_test = np.argmax(A3_test, axis = 1)
        y_test = np.array(y_test, dtype = int)
        accuracy_test = (predict_test == y_test).mean()
        Accuracy_test.append(accuracy_test)

        #loss
        Loss_test.append(loss_test)

        print("Epoch : ", epoch+1, "Test_Accuracy : ", accuracy_test, "Test_Loss : ", loss_test)
        print("\n")

    #visualizing

    # Accuracy
    plt.plot(Accuracy, color = 'm')
    plt.xlabel('Accuracy_train')
    plt.ylabel('Epoch')
    plt.show()

    # Loss
    plt.plot(Loss, color = 'm')
    plt.xlabel('Loss_train')
    plt.ylabel('Epoch')
    plt.show()

    # Accuracy_test
    plt.plot(Accuracy_test, color = 'm')
    plt.xlabel('Accuracy_test')
    plt.ylabel('Epoch')
    plt.show()

    # Loss_test
    plt.plot(Loss_test, color = 'm')
    plt.xlabel('Loss_test')
    plt.ylabel('Epoch')
    plt.show()

    # visualizing Mnist data & label

    # one-hot-encoding for Y data
    Y_hot_test = np.array(y_test, dtype = int)
    Y_hot_test = np.eye(10)[Y_hot_test]
    Y_hot_test = np.array(Y_hot_test, dtype = float)

    #feed forward

    # input layer -- hidden layer1 
    Z1_test = input_to_hidden1.forward(X_test)
    A1_test = Activation_func.forward(Z1_test)

    # hidden layer1 -- hidden layer2
    Z2_test = hidden1_to_hidden2.forward(A1_test)
    A2_test = Activation_func.forward(Z2_test)

    # hidden layer2 -- output layer
    Z3_test = hidden2_to_output.forward(A2_test)
    A3_test, loss_test = Softmax_with_loss.forward(Z3_test, Y_hot_test)


    #predict
    predict_test = np.argmax(A3_test, axis = 1)
    y_test = np.array(y_test, dtype = int)
    accuracy_test = (predict_test == y_test).mean()
    Accuracy_test.append(accuracy_test)

    for i in range(10):
        random_index = np.random.randint(0,X_test.shape[0])
        plt.imshow(X_test[random_index].reshape((28,28)), cmap='gray')
        plt.show()

        print("Predict[",i+1,"] : ", predict_test[random_index])
        print("Correct[",i+1,"] : ", y_test[random_index])
        print("\n")


mnist = fetch_openml('mnist_784')

# split dataset
X_train, y_train, X_test, y_test = split_dataset(mnist, ratio_test = 0.1)

#start training

#print("Single layer!\n\n")
#SLP(X_train, y_train, X_test, y_test, n_class_in = 10, n_epoch_in = 100, batch_size_in = 100, learning_rate_in = 0.1)

print("\n\nMulti layer!")
MLP(X_train, y_train, X_test, y_test, n_class_in = 10, n_epoch_in = 200, batch_size_in = 100, learning_rate_in = 0.1)


# %%