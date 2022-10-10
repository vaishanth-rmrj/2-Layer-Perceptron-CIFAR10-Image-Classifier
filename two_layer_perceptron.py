import numpy as np

class TwoLayerPerceptron:
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(self,
        input_dim = 3 * 32 * 32,
        hidden_dim = 100,
        num_classes = 10,
        weight_scale = 1e-3,
        reg=0.0):

        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # each column in weight matrix depicts a perceptron
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """        
        W1, W2 = self.params['W1'], self.params['W2'] # initializing weights        
        b1, b2 = self.params['b1'], self.params['b2'] # initializing biases
        
        # forward pass
        z1, relu1_cache = self.affine_relu_forward(X, W1, b1) #layer 1
        scores, relu2_cache = self.affine_relu_forward(z1, W2, b2) #layer 2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, gradients = 0, {}

        # using softmax to compute loss and gradient of output layer
        loss, grad = self.softmax_loss(scores, y)
        # applying regularization to loss
        loss += 0.5 * self.reg * (np.sum(W1*W1)+np.sum(W2*W2))
        
        # computing gradients
        upstream_gradient = grad
        # layer 2
        dx2, dw2, db2 = self.affine_relu_backward(upstream_gradient, relu2_cache)
        # layer 1
        dx, dw, db = self.affine_relu_backward(dx2, relu1_cache)
        # regularizing and storing gradients for weights and biases
        gradients['W2'] = dw2 + self.reg * W2
        gradients['b2'] = db2
        gradients['W1'] = dw + self.reg * W1
        gradients['b1'] = db
       
        return loss, gradients

    # forward pass functions

    def affine_forward(self, x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        Inputs:
        - x: np array containing input data, of shape (N, d_1, ..., d_k)
        - w: np array of weights, of shape (D, M)
        - b: np array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """        
        x = x.reshape(x.shape[0], -1) # reshaping inputs into rows        
        scores = x.dot(w) + b # adding bias to dot product of inputs and weights
        cache = (x, w, b)

        return scores, cache

    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        
        out = np.maximum(0, x) # relu activation
        cache = x

        return out, cache

    
    def affine_relu_forward(self, x, w, b):
        """
        Convenience layer that perorms an affine transform followed by a ReLU

        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer

        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        scores, fc_cache = self.affine_forward(x, w, b)
        out, relu_cache = self.relu_forward(scores)
        cache = (fc_cache, relu_cache)

        return out, cache
    
    def affine_backward(self, dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
            - x: Input data, of shape (N, d_1, ... d_k)
            - w: Weights, of shape (D, M)
            - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache        
        x = x.reshape(x.shape[0], -1) # reshaping inputs into rows
        # gradient for inputs
        dx = np.dot(dout, w.T)
        dx = dx.reshape(x.shape)
        
        dw = np.dot(x.T, dout) # weight gradient        
        db = dout.sum(axis=0) # bias gradient

        return dx, dw, db
    
    def relu_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        x = cache        
        # we keep the upstream gradient for all x > 0
        dx = dout * (x > 0)
        
        return dx

    def affine_relu_backward(self, dout, cache):
        """
        Backward pass for the affine-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = self.relu_backward(dout, relu_cache)
        dx, dw, db = self.affine_backward(da, fc_cache)
        return dx, dw, db

    def svm_loss(self, x, y):
        """
        Computes the loss and gradient using for multiclass SVM classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        num_train = x.shape[0]
        # assiging x to scores
        scores = x
        # fetching correct class scores and reshaping
        correct_class_score = scores[np.arange(num_train), y].reshape(-1, 1)
        # computing svm loss margin adn setting true class margin to 0
        margin = np.maximum(0, scores - correct_class_score + 1.0)
        margin = np.where(margin == 1.0, 0, margin)

        # averaging loss
        loss = np.sum(margin)
        loss /= num_train

        # creating gradient mask using scores
        margin_mask = np.zeros_like(margin)
        margin_mask[margin> 0] = 1
        # setting correct class gradient to neg sum of other class 1's
        margin_mask[np.arange(num_train), y] -= np.sum(margin_mask, axis = 1)
        dx = margin_mask
        dx /= num_train  

        return loss, dx

    def softmax_loss(self, x, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """        
        num_train = x.shape[0]
        # assiging x to scores
        scores = x
        # computing probabilities for different class scores
        probs = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
        # computibng loss for true class
        loss = -np.log(probs[np.arange(num_train), y])
        loss = np.sum(loss)
        # averaging loss
        loss /= num_train

        # computing gradients
        d_scores = probs.reshape((num_train, -1))
        # setting score change for true class to be zero
        d_scores[np.arange(num_train), y] -= 1
        # averaging gradients
        dx = d_scores / num_train
        
        return loss, dx
    

    
    


