import numpy as np

def discrepancy_loss(c, x, y, x_control, alpha, b, K, sensitive_attrs):
	svm_loss = 0.0
	coloring_loss = 0.0
	# assert no of samples
	for i in range(x.shape[0]):
		for i in range(y.shape[0]):
			svm_loss += 0.5*y[i]*y[j]*c[i]*c[j]*K(x[i],x[j])
	svm_loss -= c.sum()

	for attr in sensitive_attrs:
		if(attr=="sex"):
			cond = x_control[attr] == 1
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