import numpy as np

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class TwoLayerNet(object):
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

    def __init__(self, input_dim=3*32*32, hidden_dims=100, num_classes=10,
                   dropout=0, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dims: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize W1, W2, b1, and b2.  Store these as self.params['W1'], 
        #   self.params['W2'], self.params['b1'] and self.params['b2']. The
        #   biases are initialized to zero and the weights are initialized
        #   so that each parameter has mean 0 and standard deviation weight_scale.
        #   The dimensions of W1 should be (input_dim, hidden_dim) and the
        #   dimensions of W2 should be (hidden_dims, num_classes)
        # ================================================================ #

        self.params['W1'] = weight_scale * np.random.randn(hidden_dims, input_dim).T
        self.params['b1'] = np.zeros(hidden_dims)
        self.params['W2'] = weight_scale * np.random.randn(num_classes, hidden_dims).T
        self.params['b2'] = np.zeros(num_classes)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

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
        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the two-layer neural network. Store
        #   the class scores as the variable 'scores'.  Be sure to use the layers
        #   you prior implemented.
        # ================================================================ #
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        
        h1, cache = affine_forward(X, W1, b1)
        h1, cache = relu_forward(h1)
        h2, cache = affine_forward(h1, W2, b2)
        scores = h2
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the two-layer neural net.  Store
        #   the loss as the variable 'loss' and store the gradients in the 
        #   'grads' dictionary.  For the grads dictionary, grads['W1'] holds
        #   the gradient for W1, grads['b1'] holds the gradient for b1, etc.
        #   i.e., grads[k] holds the gradient for self.params[k].
        #
        #   Add L2 regularization, where there is an added cost 0.5*self.reg*W^2
        #   for each W.  Be sure to include the 0.5 multiplying factor to 
        #   match our implementation.
        #
        #   And be sure to use the layers you prior implemented.
        # ================================================================ #    

        
        loss, dout = softmax_loss(h2, y)
        loss += self.reg*(np.sum(W1**2) + np.sum(W2**2))/2
        
        dx, dw, db = affine_backward(dout, cache)
        
        grads['b2'] = db
        grads['W2'] = dw + self.reg*W2
        dLdb1 = relu_backward(dx, h1)
        grads['b1'] = np.sum(dLdb1, axis = 0)
        
        cache = (X, W1, b1)
        dx, dw, db = affine_backward(dLdb1, cache)
        
        grads['W1'] = dw + self.reg*W1
        

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize all parameters of the network in the self.params dictionary.
        #   The weights and biases of layer 1 are W1 and b1; and in general the 
        #   weights and biases of layer i are Wi and bi. The
        #   biases are initialized to zero and the weights are initialized
        #   so that each parameter has mean 0 and standard deviation weight_scale.
        # ================================================================ #

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims)-1):
            self.params['W{}'.format(i+1)] = weight_scale * np.random.randn(dims[i+1], dims[i]).T
            self.params['b{}'.format(i+1)] = np.zeros(dims[i+1])
            if use_batchnorm and i != len(dims)-2:
                self.params['gamma{}'.format(i+1)] = np.ones(dims[i+1])
                self.params['beta{}'.format(i+1)] = np.zeros(dims[i+1])
        

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode   
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the FC net and store the output
        #   scores as the variable "scores".
        # ================================================================ #

        # Unpack variables from the params dictionary
        weight = []
        bias = []
        gamma = []
        beta = []
        keys = []
        bn_caches = []
        dropout_caches = []

        for key in self.params.keys():
            keys.append(key)
            if key[0] == 'W':
                weight.append(self.params[key])
            elif key[0] == 'b' and len(key) < 4:
                bias.append(self.params[key])
            elif key[0] == 'g':
                gamma.append(self.params[key])
            elif key[0] == 'b':
                beta.append(self.params[key])
                
            
        N = X.shape[0]
        
        hiddens = [X]
        for i in range(len(weight)-1):
            h_temp, cache = affine_forward(hiddens[i], weight[i], bias[i])
            
            if self.use_batchnorm:
                h_temp, cache = batchnorm_forward(h_temp, gamma[i], beta[i], self.bn_params[i])
                bn_caches.append(cache)
                
            h_temp, cache = relu_forward(h_temp)
            
            if self.use_dropout:
                h_temp, cache = dropout_forward(h_temp, self.dropout_param)
                dropout_caches.append(cache)
            
            hiddens.append(h_temp)
            
            
        h_temp, cache = affine_forward(hiddens[i+1], weight[i+1], bias[i+1])
        scores = h_temp
        hiddens.append(h_temp)
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backwards pass of the FC net and store the gradients
        #   in the grads dict, so that grads[k] is the gradient of self.params[k]
        #   Be sure your L2 regularization includes a 0.5 factor.
        # ================================================================ #
        multiplier = 2
        dec = 0
        if self.use_batchnorm:
            multiplier += 2
            dec += 2
        
        loss, dout = softmax_loss(hiddens[-1], y)
        for W in weight:
            loss += self.reg*np.sum(W**2)/2

        dx, dw, db = affine_backward(dout, cache)
        
        grads[keys[-1]] = db
        grads[keys[-2]] = dw + self.reg*weight[-1]
        
        
        for i in range(len(weight) - 1):
            
            if self.use_dropout:
                dx = dropout_backward(dx, dropout_caches[-1-i])
            
            dLdb = relu_backward(dx, hiddens[-i-2])
            
            
            if self.use_batchnorm:
                dLdb, dgamma, dbeta = batchnorm_backward(dLdb, bn_caches[-1-i])
                grads[keys[-i*multiplier-1-dec]] = dbeta
                grads[keys[-i*multiplier-2-dec]] = dgamma
            
            grads[keys[-i*multiplier-3-dec]] = np.sum(dLdb, axis = 0)

            
            
            cache = (hiddens[-i-3], weight[-i-2], bias[-i-2])
            dx, dw, db = affine_backward(dLdb, cache)

            grads[keys[-i*multiplier-4-dec]] = dw + self.reg*weight[-i-2]

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grads
