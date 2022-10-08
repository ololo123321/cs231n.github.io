from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        xi = X[i]
        scores = xi.dot(W)
        m = scores.max()
        scores_exp = np.exp(scores - m)
        scores_exp_sum = scores_exp.sum()
        probs = scores_exp / scores_exp_sum
        loss += -np.log(probs[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += xi * (probs[j] - 1.0)
            else:
                dW[:, j] += xi * probs[j]
    loss /= num_train
    loss += reg * np.square(W).sum()
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    # loss = 0.0
    # dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X @ W  # [N, C]
    m = scores.max(1, keepdims=True)
    scores_exp = np.exp(scores - m)
    scores_exp_sum = scores_exp.sum(1, keepdims=True)
    probs = scores_exp / scores_exp_sum
    correct_class_idx = np.s_[range(num_train), y]
    loss = -np.log(probs[correct_class_idx]).mean()

    y_oh = np.zeros_like(probs)
    y_oh[correct_class_idx] = 1.0
    dW = X.T @ (probs - y_oh)  # [D, C]
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
