import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = int(1 + (H + 2*pad - HH)/stride)
    W_out = int(1 + (W + 2*pad - WW)/stride)
    out = np.zeros((N, F, H_out, W_out))
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad),(pad, pad)), constant_values = 0)
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] = np.sum(x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]*w[f, :, :, :]) + b[f]
                    
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param['stride'], conv_param['pad']]
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #
    
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis = (0, 2, 3))
    
    n, c, hh, ww = dout.shape
#     x_dilated = np.zeros((n, c, stride*hh - stride + 1, stride*ww - stride + 1), dtype = x.dtype)
#     x_dilated[:, :, ::stride, ::stride] = x
    dout_dilated = np.zeros((n, c, stride*hh, stride*ww), dtype = x.dtype)
    dout_dilated[:, :, ::stride, ::stride] = dout
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad+1),(pad, pad+1)), constant_values = 0)
    
    N, C, H, W = x_padded.shape
    
    
    for n in range(N):
        for f in range(F):
            for d in range(C):
                for i in range(W - out_width*stride + stride - 1):
                    for j in range(H - out_height*stride + stride - 1):
                        
                        dw[f, d, i, j] += np.sum(dout_dilated[n, f, :, :]*x_padded[n, d, i:i+out_width*stride, j:j+out_height*stride])
    
    _, _, H, W = x.shape
    _, _, H_w, W_w = w.shape
    w_rotated = np.rot90(w, axes = (2, 3))
    w_rotated = np.rot90(w_rotated, axes = (2, 3))
    dout_padded = np.pad(dout_dilated, ((0, 0), (0, 0), (pad, pad),(pad, pad)), constant_values = 0)
    
    
    N, C, H, W = x.shape
    for n in range(N):
        for f in range(F):
            for d in range(C):
                for i in range(W):
                    for j in range(H):
                        dx[n, d, i, j] += np.sum(dout_padded[n, f, i:i+W_w, j:j+H_w]*w_rotated[f, d, :, :])
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #
    
    pool_width, pool_height, stride = pool_param['pool_width'], pool_param['pool_height'], pool_param['stride']
    N, C, H, W = x.shape
    out_height = int((H-pool_height)/stride + 1)
    out_width = int((W-pool_width)/stride + 1)
    out = np.zeros((N, C, out_height, out_width))
    x_temp = x
    
    for n in range(N):
        for c in range(C):
            for i in range(out_height):
                for j in range(out_width):
                    out[n, c, i, j] = np.max(x_temp[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #
    
    pool_width, pool_height, stride = pool_param['pool_width'], pool_param['pool_height'], pool_param['stride']
    N, C, H, W = x.shape
    N, C, H_dout, W_dout = dout.shape
    dx = x
    dout_temp = dout
    
    for n in range(N):
        for c in range(C):
            for i in range(H_dout):
                for j in range(W_dout):
                    local_max = np.max(dx[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
                    dx_local = dx[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    dx_local[dx_local<local_max] = 0
                    dx_local[dx_local != 0] = 1
                    dx_local = dx_local*dout_temp[n, c, i, j]
                    dx[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] = dx_local
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    
    N, C, H, W = x.shape
    x_temp = np.zeros((N*H*W, C))
    for c in range(C):
        x_temp[:, c] = np.reshape(x[:,c,:,:], (N*H*W))
    
    out_temp, cache = batchnorm_forward(x_temp, gamma, beta, bn_param)
    
    out = np.zeros((N, C, H, W))
    for c in range(C):
        out[:,c,:,:] = np.reshape(out_temp[:,c], (N, H, W))
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    
    N, C, H, W = dout.shape
    x, gamma, var, mu, eps, mode, momentum = cache
    x_temp = x
    dout_temp = np.zeros((N*H*W, C))
    for c in range(C):
        dout_temp[:, c] = np.reshape(dout[:,c,:,:], (N*H*W))
    
    cache_temp = x_temp, gamma, var, mu, eps, mode, momentum
    
    dx_temp, dgamma, dbeta = batchnorm_backward(dout_temp, cache_temp)
    
    dx = np.zeros((N, C, H, W))
    for c in range(C):
        dx[:,c,:,:] = np.reshape(dx_temp[:,c], (N, H, W))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx, dgamma, dbeta