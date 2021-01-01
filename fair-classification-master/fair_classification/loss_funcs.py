import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy



def _hinge_loss(w, X, y):

    
    yz = y * np.dot(X,w) # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1-yz)) # hinge function
    
    return sum(yz)

def _logistic_loss(w, X, y, return_arr=None):
	"""Computes the logistic loss.

	This function is used from scikit-learn source code

	Parameters
	----------
	w : ndarray, shape (n_features,) or (n_features + 1,)
	    Coefficient vector.

	X : {array-like, sparse matrix}, shape (n_samples, n_features)
	    Training data.

	y : ndarray, shape (n_samples,)
	    Array of labels.

	"""
	

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	if return_arr == True:
		out = -(log_logistic(yz))
	else:
		out = -np.sum(log_logistic(yz))
	return out

def _logistic_loss_l2_reg(w, X, y, lam=None):

	if lam is None:
		lam = 1.0

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	logistic_loss = -np.sum(log_logistic(yz))
	l2_reg = (float(lam)/2.0) * np.sum([elem*elem for elem in w])
	out = logistic_loss + l2_reg
	return out


def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0

	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function

	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----

	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

def discrepancy_loss(x, y, c, alpha, b, K, sensitive_attrs):
	svm_loss = 0.0
	coloring_loss = 0.0
	#assert no of samples
	for i in range(x.shape[0]):
		for i in range(y.shape[0]):
			svm_loss += 0.5*y[i]*y[j]*c[i]*c[j]*K(x[i],x[j])
	svm_loss -= c.sum()

	for attr in sensitive_attrs:
		if(attr=="sex"):
			cond = x[attr] == "Male"
			male_x = x[cond]
			male_y = y[cond]
			female_x = x[~cond]
			female_y = y[~cond]
			male_loss  = 0.0
			female_loss = 0.0

			for i in range(male_x.shape[0]):
				temp=0.0
				for j in range(x.shape[0]):
					temp+= c[j]*y[j]*K(x[i], male_x[i])
				temp-=b
				male_loss += np.tanh(temp)

			for i in range(female_x.shape[0]):
				temp=0.0
				for j in range(x.shape[0]):
					temp+= c[j]*y[j]*K(x[i], female_x[i])
				temp-=b
				female_loss += np.tanh(temp1)

			coloring_loss = max(abs(male_loss), abs(female_loss))

	loss = (1-alpha)*svm_loss + alpha*coloring_loss
	return loss	