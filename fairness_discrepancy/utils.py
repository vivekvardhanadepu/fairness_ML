import numpy as np
from loss_wrapper import loss_wrapper
from random import seed
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import _cobyla

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def rbf_kernel(x1, x2, gamma = 0.0001):
    return np.exp(-1*gamma*np.linalg.norm(x1-x2, axis=1)**2)


# def rbf_kernel_optimized(x1, x2):
#     arr = np.array((x1.shape[0], x2.shape[0]))
#     arr = []
#     for i in range(x1.shape[0]):
#         arr.append(np.squeeze(rbf_kernel(x1[i], x2)))
#     return np.array(arr)

def train_model(x, y, x_control, loss_function, sensitive_attrs, max_iter = 1000, 
                    alpha=0.5, l=1, method='cobyla', initiator=0.0001, catol=0.1,
                        batches=100):

    """

    Function that trains the model with fairness constraints.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control

    ----

    Outputs:

    c: the learned weight vector for the classifier

    """
#     _cobyla.minimize(get_one_hot_encoding, m=x, x=np.zeros(4, np.float64), rhobeg=2.5,
#                                   rhoend=3.5, iprint=1, maxfun=20,
#                                   dinfo=np.zeros(4, np.float64))

    n = x.shape[0]
    x_init = np.random.rand(n, )*initiator
    print("iter: ", max_iter, ", lambda: ", l, ", alpha: ", alpha, ", kernel: rbf"\
            " method: ", method, ", catol: ", catol, "batches: ", batches)
    print("x_init: ", x_init)
    c = x_init
    
    kernel = rbf_kernel
    kernel_values = []
    
    for i in range(x.shape[0]):
        kernel_values.append(np.squeeze(kernel(x[i], x)))
        
    kernel_matrix  = np.array(kernel_values)
    
    f_args=(x, y, x_control, alpha, kernel, kernel_matrix, sensitive_attrs)
    kernel_obj = loss_wrapper(loss_function)
    constraints = []

    def lambda_contraint(c, i, n, l):
        return c[i]-(1/(2*n*l))
    
    def non_neg_constraint(c, i):
        return -1*c[i]
    
    def dot_y_constraint(c, y, tol):
        return np.dot(c, y) - tol

    def dot_y_constraint_neg(c, y, tol):
        return -tol -np.dot(c, y)

    for i in range(n):
        c1 = ({'type': 'ineq', 'fun': lambda_contraint, 'args':(i, n, l)})
        c2 = ({'type': 'ineq', 'fun': non_neg_constraint, 'args': (i,)})
        constraints.append(c1)
        constraints.append(c2)

    c3 = ({'type': 'ineq', 'fun': dot_y_constraint, 'args': (y, catol)})
    c4 = ({'type': 'ineq', 'fun': dot_y_constraint_neg, 'args': (y, catol)})
    constraints.append(c3)
    constraints.append(c4)

    #constraints.append(LinearConstraint(y, lb=0.1, ub=0.1))
    c = minimize(fun = kernel_obj.simulate,
        x0 = c,
        args = f_args,
        method = method,
        options = {"maxiter":max_iter},
        #bounds = [(0, 1/(2*n*l)) for i in range(n)]
        constraints = constraints
        )
    
    print
    print("weights: ", c.x)
    print("cy dot constraint :", np.dot(c.x,y))
    print
    
    try:
        assert(c.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(c)
        print

    return c, kernel_obj.values['c'], kernel_obj.values['losses']

def compute_p_rule(x_control, class_labels):

    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1.0) # non-protected group
    prot_all = sum(x_control == 0.0) # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0
    print
    print("Total data points: %d" % (len(x_control)))
    print("# non-protected examples: %d" % (non_prot_all))
    print("# protected examples: %d" % (prot_all))
    print("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, frac_non_prot_pos * 100.0))
    print("Protected in positive class: %d (%0.0f%%)" % (prot_pos, frac_prot_pos * 100.0))
    # print("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    # print("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    print("P-rule is: %0.0f%%" % ( p_rule ))
    return p_rule
    
def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """
    for k in in_arr:
        if str(type(k)) != "<class 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
            return None
        
    in_arr = np.array(in_arr, dtype=int)
    assert(len(in_arr.shape)==1) # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    
    index_dict = {} # value to the column number
    for i in range(0,len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []    
    for i in range(0,len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1 # set that value of tuple to 1
        out_arr.append(tup)

    return (np.array(out_arr), index_dict)

def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):

    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = {}
    x_control_all_test = {}
    for k in x_control_all.keys():
        x_control_all_train[k] = x_control_all[k][:split_point]
        x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test

def predict(model, x_train, y_train, x_test):

    K = rbf_kernel
    n = x_train.shape[0]
    m = x_test.shape[0]
    y_train_predicted = []
    y_test_predicted = []

    # using 0th element to get b
    # b = 0.0
    # for i in range(n):
    #     b += model[i]*y_train[i]*K(x_train[0], x_train[i])
    b = np.dot(model*y_train, np.squeeze(K(x_train[0], x_train)))
    b -= y_train[0]
    
    for j in range(n):
        temp = 0.0
        # for i in range(n):
        #     temp+= model[i]*y_train[i]*K(x_train[j], x_train[i])
        temp = np.dot(model*y_train, K(x_train[j], x_train))
        temp -= b
        y_train_predicted.append(np.sign(temp))

    for j in range(m):
        temp = 0.0
        # for i in range(n):
        #     temp+= model[i]*y_train[i]*K(x_test[j], x_train[i])
        temp = np.dot(model*y_train, K(x_test[j], x_train))
        temp -= b
        y_test_predicted.append(np.sign(temp))

    y_test_predicted = np.array(y_test_predicted)
    y_train_predicted = np.array(y_train_predicted)

    return y_train_predicted, y_test_predicted

def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):


    """
    returns the train/test accuracy of the model
    we either pass the model (w)
    else we pass y_predicted
    """
    if model is not None and y_test_predicted is not None:
        print("Either the model (w) or the predicted labels should be None")
        raise Exception("Either the model (w) or the predicted labels should be None")

    # subtract b
    if model is not None:
        y_train_predicted, y_test_predicted = predict(model, x_train, y_train, x_test)

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test