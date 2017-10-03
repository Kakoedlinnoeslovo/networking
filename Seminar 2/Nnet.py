if __name__ == 'main':
    import numpy as np
    from numpy.random import seed as sd
    sd = 42
else:
    import numpy as np
    from numpy.random import seed as sd
    sd = 42

class Linear:

    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).

        self.w = np.random.normal(0, 0.1, (input_size + 1, output_size))

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        self.X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        self.y = np.dot(self.X, self.w)
        return self.y

    def backward(self, dLdy):
        '''
        dLdy [N, output_size]
        '''
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        self.dLdw = np.dot(self.X.T, dLdy)
        self.dLdx = np.dot(dLdy, self.w.T)

        return self.dLdx[:, :(self.X.shape[1] - 1)]

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        #### YOUR CODE HERE

        self.w = self.w - learning_rate * self.dLdw


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        self.X = np.array(X)
        self.sig = 1 / (1 + np.exp(-self.X))
        return self.sig

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE

        return dLdy * self.sig * (1 - self.sig)

    def step(self, learning_rate):
        pass

    

class ReLU:
    def __init__(self):
        pass

    def forward(self, X):
        return np.maximum(0, X)

    def backward(self, X):
        epsilon=0.1
        gradients = 1. * (X > 0)
        gradients[gradients == 0] = epsilon
        return gradients

    def step(self, learning_rate):
        pass  
    
class eLU:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
        pass

    def forward(self, X):
        return np.where(X > 0, X, self.alpha*(np.exp(X)-1))

    def backward(self, X):
        return np.where(X > 0, 1, self.alpha*(np.exp(X)-1)+self.alpha)

    def step(self, learning_rate):
        pass    
    
    
class NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass

    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        y is np.array of size (N,), contains correct labels
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        self.X = X
        self.y = y
        Xmax = np.max(self.X, axis=1)  # keep axis 0 (different points) intact
        self.L = self.X[range(X.shape[0]), y] - Xmax - np.log(
            # trigger NumPy's "smart indexing" by feeding both array-like
            np.sum(  # log-sum-exp to avoid floating overflows
                np.exp(
                    (self.X.T - Xmax).T  # hooray for NumPy broadcasting
                ),
                axis=1
            )
        )

        return -self.L

    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        local_max = np.max(self.X, axis=1)
        self.dLdx = -np.exp(
            self.X.T - local_max - np.log(
                np.sum(
                    np.exp(self.X.T - local_max),
                    axis=0
                )
            )
        ).T
        self.dLdx[range(self.X.shape[0]), self.y] += 1
        self.dLdx = -self.dLdx  # *negative*

        return self.dLdx


class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE

        self.list = []
        for m in modules:
            self.list.append(m)

    def forward(self, X):
        #### YOUR CODE HERE
        #### Apply layers to input
        XX = np.array(X)
        for i in self.list:
            XX = i.forward(XX)
        return XX

    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        #### YOUR CODE HERE
        QQ = dLdy
        for i in self.list[::-1]:
            QQ = i.backward(QQ)
        return QQ

    def step(self, learning_rate):
        for l in self.list:
            l.step(learning_rate)