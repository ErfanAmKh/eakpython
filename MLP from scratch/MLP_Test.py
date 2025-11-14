import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

data = load_breast_cancer()

class MLP:
    def __init__(self, _alpha, _Dataset, _Labels, _number_of_neurons, _Epoch, _Tolerance):
        #Attributes
        self.alpha = _alpha
        self.X = _Dataset
        self.y = _Labels
        self.K = _number_of_neurons
        self.epochs = _Epoch
        self.tolerance = _Tolerance

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        self.n = self.X_train.shape[1]
        self.p = self.X_train.shape[0]

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.X_train = np.c_[self.X_train, np.ones(self.p)]
        self.X_test = np.c_[self.X_test, np.ones(self.X_test.shape[0])]
        self.n += 1

        print(self.X_train[2])

        # print("n: ", self.n)
        # print("p: ", self.p)
        # print("number of samples: ", self.X.shape)

    def initialize_weights(self):
        _v = np.random.randn(self.n, self.K) * 100
        temp_y = np.expand_dims(self.y, axis=1)
        _w = np.random.randn(self.K, temp_y.shape[1]) * 0.0001
        return (_v, _w)
    
    def sigmoid(self, _input):
        return (1)/(1 + np.exp(-_input))
    
    def f(self, _input, _weihgts):
        z = np.dot(_input, _weihgts)
        return self.sigmoid(z)
    
    def train(self):
        V, W = self.initialize_weights()

        Error = []
        for i in range(self.epochs):
            E = 0
            for j in range(self.p):
                x = self.X_train[j]
                # print("x: ", x.shape)
                # input('hey')
                d = self.y_train[j]
                # print("d: ", d)
                # input('hey')
                o1 = self.f(x, V)
                # print("o1: ", o1)
                # input('hey')
                o = self.f(o1, W)
                # print("o: ", o)
                # input('hey')
                E += (d - o) ** 2
                # print("E: ", E)
                # input('hey')
                delta_2 = (d - o) * o * (1- o)
                # print("delta 2: ", delta_2)
                # input('hey')
                # print("o1 shape: ", o1.shape, " (1-o1) shape: ", (1-o1).shape, " W shape: ", W.shape, " delta 2 shape: ", delta_2.shape)
                delta_1 = np.expand_dims(o1, axis=1) * np.expand_dims(1 - o1, axis=1) * W * delta_2
                # print("delta 1: ", delta_1)
                # input('hey')
                W += self.alpha * delta_2 * np.expand_dims(o1, axis=1)
                # W += self.alpha * np.outer(o1, delta_2)
                # print("W shape: ", W.shape)
                # input('hey')
                V += self.alpha * np.outer(x, delta_1)
                # print("V shape: ", V.shape)
                # input('hey')
            Error.append(E)
            print("Epoch:", i, "  Loss: ", E)
            if E / self.p < self.tolerance:
                print("Converged at epoch number: ", i)
                break
        return (V, W, Error)
    
    def validate(self):
        v, w, e = self.train()


        correct_detection = 0
        for i in range(self.p):
            x = self.X_train[i]
            d = self.y_train[i]
            o1 = self.f(x, v)
            o = self.f(o1, w)
            prediction = 0
            if o > 0.5:
                prediction = 1
            if prediction == d:
                correct_detection += 1
        
        train_acc = 100 * (correct_detection / self.p)
        print("Train Accuracy: ", train_acc)

        correct_detection = 0
        for i in range(len(self.X_test)):
            x = self.X_test[i]
            d = self.y_test[i]
            o1 = self.f(x, v)
            o = self.f(o1, w)
            prediction = 0
            if o > 0.5:
                prediction = 1
            if prediction == d:
                correct_detection += 1
        
        test_acc = 100 * (correct_detection / len(self.X_test))
        print("Test Accuracy: ", test_acc)
        
        plt.figure()
        plt.plot(e, 'r')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Sum of square errors over training set per epoch")
        plt.grid()
        plt.show()




mlp = MLP(0.1, data.data, data.target, 50, 1000, 0.01)

mlp.validate()
