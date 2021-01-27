import utils as ut
import loss_funcs as lf
from prep_adult_data import *

def test_adult_data():
	

	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=10000) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data



	""" Split the data into train and test """
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	train_fold_size = 0.7
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	loss_function = lf.discrepancy_loss
	sensitive_attrs = ["sex"]

	def train_test_classifier():
		w = ut.train_model(x_train, y_train, x_control_train, loss_function, sensitive_attrs)
		train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
		distances_boundary_test = (np.dot(x_test, w)).tolist()
		all_class_labels_assigned_test = np.sign(distances_boundary_test)
		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
		return w, p_rule, test_score

	""" Classify the data while optimizing for accuracy """
	w_uncons, p_uncons, acc_uncons = train_test_classifier()

	return

def main():
	test_adult_data()


if __name__ == '__main__':
	main()