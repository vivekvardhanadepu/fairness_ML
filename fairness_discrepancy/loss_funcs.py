import numpy as np

def discrepancy_loss(c, x, y, x_control, alpha, K, sensitive_attrs):
	svm_loss = 0.0
	coloring_loss = 0.0
	# assert no of samples
	# for i in range(x.shape[0]):
	# 	for j in range(y.shape[0]):
	# 		svm_loss += 0.5*y[i]*y[j]*c[i]*c[j]*K(x[i],x[j])
	kernel_values = []

	for i in range(x.shape[0]):
		kernel_values.append(np.squeeze(K(x[i], x)))
	
	kernel_values  = np.array(kernel_values)
	svm_loss = .5*np.dot((c*y).T @ kernel_values, c*y)
	svm_loss -= c.sum()

	for attr in sensitive_attrs:
		if(attr=="sex"):
			cond = x_control[attr] == 1.0
			male_y = y[cond]	
			female_y = y[~cond]
			male_loss  = male_y.sum()
			female_loss = female_y.sum()

			# for i in range(male_x.shape[1]):
			# 	temp=0.0
			# 	for j in range(x.shape[1]):
			# 		temp+= c[j]*y[j]*K(x[j], male_x[i])
			# 	temp-=b
			# 	male_loss += np.tanh(temp)

			coloring_loss = max(abs(male_loss), abs(female_loss))

	loss = (1-alpha)*svm_loss + alpha*coloring_loss
	print("loss: ", loss)
	return loss	