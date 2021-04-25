import numpy as np
from numba import jit

# @jit(nopython=True, parallel=True)
def discrepancy_loss(c, x, y, x_control, alpha, kernel_matrix, sensitive_attrs):
    svm_loss = 0.0
    coloring_loss = 0.0
    # assert no of samples
    temp_matrix = (c*y).T @ kernel_matrix
    svm_loss = .5*np.dot(temp_matrix, c*y)
    svm_loss -= c.sum()

    b = (temp_matrix.sum() - y.sum())/x.shape[0]

    for attr in sensitive_attrs:
        if(attr=="sex"):
            cond = x_control[attr] == 1.0
            # male_y = y[cond]	
            # female_y = y[~cond]
            male_kernel_matrix = kernel_matrix[cond]
            female_kernel_matrix = kernel_matrix[~cond]
            male_loss  = 0.0
            female_loss = 0.0

            male_loss = np.tanh(((c*y).T @ male_kernel_matrix.T) - b).sum()
            female_loss = np.tanh(((c*y).T @ female_kernel_matrix.T) - b).sum()
            # for i in range(male_kernel_matrix.shape[0]):
            #     temp = np.dot(c*y, male_kernel_matrix[i]) - b
            #     male_loss += np.tanh(temp)
                
            # for i in range(female_kernel_matrix.shape[0]):
            #     temp = np.dot(c*y, female_kernel_matrix[i]) - b
            #     female_loss += np.tanh(temp)
                
            coloring_loss = max(abs(male_loss), abs(female_loss))

    loss = (1-alpha)*svm_loss + alpha*coloring_loss
    print("alpha: ", alpha)
    print("c: ", c)
    print("loss: ", loss)
    return loss	