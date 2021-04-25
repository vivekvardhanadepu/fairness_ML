import utils as ut
import loss_funcs as lf
from prep_adult_data import *

def test_adult_data():
	

	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=10000) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data



	""" Split the data into train and test """
	train_fold_size = 0.7
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	loss_function = lf.discrepancy_loss
	sensitive_attrs = ["sex"]

	def train_test_classifier():
		final_c, Cs, losses, kernel_matrix = ut.train_model(x_train, y_train, x_control_train, loss_function, sensitive_attrs)
		y_train_predicted, y_test_predicted = ut.predict(final_c, x_train, y_train, x_test, kernel_matrix)
		train_score, test_score, correct_answers_train, correct_answers_test = \
						ut.check_accuracy(None, kernel_matrix, x_train, y_train, x_test, y_test, \
											y_train_predicted, y_test_predicted)
		print("Train data:")
		print("------------")
		print("Train accuracy : ", train_score)
		p_rule_train = ut.compute_p_rule(x_control_train["sex"], y_train_predicted)
		print()
		print("Test data: ")
		print("------------")
		print("Test accuracy : ", test_score)
		p_rule_test = ut.compute_p_rule(x_control_test["sex"], y_test_predicted)
		print
		print(losses)
		# return w, p_rule_train, p_rule_test

	""" Classify the data while optimizing for accuracy """
	train_test_classifier()

def main():
	test_adult_data()


if __name__ == '__main__':
	main()