import numpy as np
def unit_activation(f):
    """sumary_line
    
    Keyword arguments:
    argument f is the output of f(x) = W.X + b
    Return: the step acivation of the vector 
    mathematicaly g(x) = {1 if x >= 0 , 0 ow}
    """
    return np.where(f>=0, 1, 0)
class perceptron:
    def __init__(self, eta = 0.01, epotchs = 1000):
        self.eta = eta
        self.epotchs = epotchs
    def predict(self, xi):
        xi = np.array(xi)
        f = np.dot(xi, self.W[1:])+self.W[0]
        return unit_activation(f)
    def fit(self, X, y):
        """sumary_line
        
        Keyword arguments:
        X : the feature vector, y : the target vector
        function: fits the model using SGD to minimize loss
        """
        self.W = np.random.random(X.shape[1]+1)
        for i in range(self.epotchs):
            print("epotch {}\{} :".format(i, self.epotchs))
            for xi, yi in zip(X, y):
                y_hat_i = self.predict(xi)
                self.W[1:] = self.W[1:] + self.eta*(yi-y_hat_i)*xi
                self.W[0] = self.W[0] + self.eta*(yi-y_hat_i)
            print("loss : {}, weights : {}, bias : {}".format(self.eta*(yi-y_hat_i), self.W[1:], self.W[0]))
    def accuracy(self, y_hat,y_test):
        """sumary_line
        
        Keyword arguments:
        y_hat the prediction of the model
        y_test is the real values (ground truth)
        returns the accuray of the model
        """
        xor =np.array(np.logical_xor(y_test, y_hat))
        xor = np.where(xor==True, 1, 0)
        return (1 - (np.count_nonzero(xor)/len(y_test)))
            