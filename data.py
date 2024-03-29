import xlrd
import numpy as np
import statistics
from scipy.stats import linregress
from sklearn import linear_model
from matplotlib import pyplot
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics

loc = ("power_data.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

#Training data (First 7500 instances)
first_col = sheet.col_values(0,1,7501)
second_col = sheet.col_values(1,1,7501)
third_col = sheet.col_values(2,1,7501)
fourth_col = sheet.col_values(3,1,7501)
fifth_col = sheet.col_values(4,1,7501)

cols = [first_col,
	second_col,
	third_col,
	fourth_col,
	fifth_col]

#Test data (Last 2067 instances)
test1_col = sheet.col_values(0,7502, 9569)
test2_col = sheet.col_values(1,7502, 9569)
test3_col = sheet.col_values(2,7502, 9569)
test4_col = sheet.col_values(3,7502, 9569) 
test5_col = sheet.col_values(4,7502, 9569) 

test = [test1_col,
	test2_col,
	test3_col,
	test4_col,
	test5_col]

#Labels of our different inputs and output
l1 = sheet.col_values(0,0,1)
l2 = sheet.col_values(1,0,1)
l3 = sheet.col_values(2,0,1)
l4 = sheet.col_values(3,0,1)
l5 = sheet.col_values(4,0,1)

labels = [l1, l2, l3, l4, l5]

#Normalized data sets
train_norm = []
test_norm = []

#Calculates all of the basic statistics for our inputs and output. (Training Data)
print("<TRAINING DATA>")
x = 0
for i in cols:

	sum = 0
	l = labels[x]

	print(str(l[0]))
	print("Mean: " + str(statistics.mean(i)))
	print("Median: " + str(statistics.median(i)))
	print("Mode: " + str(statistics.mode(i)))
	print("Min: " + str(min(i)))
	print("Max: " + str(max(i)))
	print("Variance: " + str(statistics.variance(i)))
	print("StDev: " + str(statistics.stdev(i)))
	
	#Calulcates the correlation coefficients
	print("Corr. Coeff.: " + str(np.corrcoef(cols[4], i)) + "\n")

	#Normalizes our data
	norm_list = []
	for num in i:
		norm_list.append((num - min(i))/(max(i)-min(i)))
	train_norm.append(norm_list)
	
	x += 1


#Calculates all of the basic statistics for our inputs and output. (Test Data)
print("<TEST DATA>")
x = 0
for i in test:

	sum = 0
	l = labels[x]
	print(str(l[0]))
	print("Mean: " + str(statistics.mean(i)))
	print("Median: " + str(statistics.median(i)))
	print("Mode: " + str(max(set(i), key=i.count)))
	print("Min: " + str(min(i)))
	print("Max: " + str(max(i)))
	print("Variance: " + str(statistics.variance(i)))
	print("StDev: " + str(statistics.stdev(i)))
	
	#Calulcates the correlation coefficients
	print("Corr. Coeff.: " + str(np.corrcoef(test[4], i))[1] + "\n")

	#Normalizes our data
	norm_list = []
	for num in i:
		norm_list.append((num - min(i))/(max(i)-min(i)))
	test_norm.append(norm_list)
	
	x += 1

#Obtaining our linear regression constants 
#One input variable
r1 = linregress(train_norm[0], train_norm[4])	
print("One input variable:")
print("Slope: " + str(r1.slope))
print("Intercept: " + str(r1.intercept))

r2 = linregress(train_norm[1], train_norm[4])	
print("Slope: " + str(r2.slope))
print("Intercept: " + str(r2.intercept))

r3 = linregress(train_norm[2], train_norm[4])	
print("Slope: " + str(r3.slope))
print("Intercept: " + str(r3.intercept) + "\n")


#Two input variables
print("Two input variables:")
a = np.array([train_norm[0], train_norm[1]])
a = a.reshape((7500, 2))
r4 = linear_model.LinearRegression()
r4.fit(a, train_norm[4])
print("Coefficients: " + str(r4.coef_))
print("Intercept: " + str(r4.intercept_))

a = np.array([train_norm[0], train_norm[2]])
a = a.reshape((7500, 2))
r5 = linear_model.LinearRegression()
r5.fit(a, train_norm[4])
print("Coefficients: " + str(r5.coef_))
print("Intercept: " + str(r5.intercept_))

a = np.array([train_norm[1], train_norm[2]])
a = a.reshape((7500, 2))
r6 = linear_model.LinearRegression()
r6.fit(a, train_norm[4])
print("Coefficients: " + str(r6.coef_))
print("Intercept: " + str(r6.intercept_) + "\n")


#Three input variables
print("Three input variables:")
a = np.array([train_norm[0], train_norm[1], train_norm[2]])
a = a.reshape((7500, 3))
r7 = linear_model.LinearRegression()
r7.fit(a, train_norm[4])
print("Coefficients: " + str(r7.coef_))
print("Intercept: " + str(r7.intercept_) + "\n")

#Making predictions using Linear Regression models and calculating error.
c4 = r4.coef_	
c5 = r5.coef_	
c6 = r6.coef_	
c7 = r7.coef_	
result = []
x1 = test_norm[0]
x2 = test_norm[1]
x3 = test_norm[2]
comp = test_norm[4]
mse1 = 0
mse2 = 0
mse3 = 0
mse4 = 0
mse5 = 0
mse6 = 0
mse7 = 0
res1 = []
res2 = []
res3 = []
res4 = []
res5 = []
res6 = []
res7 = []
for i in range(0,2067):

	output1 = r1.slope * x1[i] + r1.intercept
	mse1 += (comp[i] - output1)**2
	res1.append(output1)

	output2 = r2.slope * x2[i] + r2.intercept
	mse2 += (comp[i] - output2)**2
	res2.append(output2)

	output3 = r3.slope * x3[i] + r3.intercept
	mse3 += (comp[i] - output3)**2
	res3.append(output3)

	output4 = r4.intercept_ + x1[i]*c4[0] + x2[i]*c4[1]
	mse4 += (comp[i] - output4)**2
	res4.append(output4)

	output5 = r5.intercept_ + x1[i]*c5[0] + x2[i]*c5[1]
	mse5 += (comp[i] - output5)**2
	res5.append(output5)

	output6 = r6.intercept_ + x1[i]*c6[0] + x2[i]*c6[1]
	mse6 += (comp[i] - output6)**2
	res6.append(output6)

	output7 = r7.intercept_ + x1[i]*c7[0] + x2[i]*c7[1] + x3[i]*c7[2]
	mse7 += (comp[i] - output7)**2
	res7.append(output7)

print("MSE1: " + str(mse1/2067))
print("MSE2: " + str(mse2/2067))
print("MSE3: " + str(mse3/2067))
print("MSE4: " + str(mse4/2067))
print("MSE5: " + str(mse5/2067))
print("MSE6: " + str(mse6/2067))
print("MSE7: " + str(mse7/2067) + " (Best - three input  variables)")


#Calculates R-squared error of our data
def R2(a, b):
	mean = statistics.mean(a)
	sse = 0
	sst = 0
	for i in range(0, len(a)):
		sse += (a[i] - b[i])**2
		sst += (a[i] - mean)**2
	return 1 - (sse/sst)

#Prints out the MSE and R-squared error values of each of our linear models.
print("One input variable (AT):")
print("MSE1: " + str(mse1/2067))
print("R^2 (1): " + str(R2(comp, res1)) + "\n")

print("One input variable (V):")
print("MSE2: " + str(mse2/2067)) 
print("R^2 (2): " + str(R2(comp, res2)) + "\n")

print("One input variable (AP):")
print("MSE3: " + str(mse3/2067))
print("R^2 (3): " + str(R2(comp, res3)) + "\n")

print("Two input variables (AT and V):")
print("MSE4: " + str(mse4/2067))
print("R^2 (4): " + str(R2(comp, res4)) + "\n")

print("Two input variables (AT and AP):")
print("MSE5: " + str(mse5/2067))
print("R^2 (5): " + str(R2(comp, res5)) + "\n")

print("Two input variables (V and AP):")
print("MSE6: " + str(mse6/2067))
print("R^2 (6): " + str(R2(comp, res6)) + "\n")

print("Three input variables (AT, V, and AP):")
print("MSE7: " + str(mse7/2067))
print("R^2 (7): " + str(R2(comp, res7)) + "\n")


#Nonlinear regression model (sigmoid function)
#Normalizes our output training values for sigmoid function.
z_norm = []
for i in train_norm[4]:
	if(i == 0):
		z_norm.append(0)
	else:
		check = 1/i - 1
		if(check == 0):
			z_norm.append(0)
		else:
			temp = -np.log(1/i - 1)	
			z_norm.append(temp)


#TODO: change non-linear model from sigmoid to tanh or ReLU
#Calculates the non-linear prediction and the error associated with it.
def f(z):
	a = np.array([train_norm[0], train_norm[1], train_norm[2]])
	a = a.reshape((7500, 3))
	r8 = linear_model.LinearRegression()
	r8.fit(a, z_norm)
	print("Non-linear Coefficients: " + str(r8.coef_))
	print("Non-linear Intercept: " + str(r8.intercept_) + "\n")
	coef = r8.coef_
	predicted = []
	predicted2 = []
	x1 = test_norm[0]
	x2 = test_norm[1]
	x3 = test_norm[2]
	actual = test_norm[4]
	mse = 0
	mseT = 0
	for i in range(0, 2067):
		predicted.append((coef[0]*x1[i]) + (coef[1]*x2[i]) + (coef[2]*x3[i]) + r7.intercept_)
		predicted2.append((coef[0]*x1[i]) + (coef[1]*x2[i]) + (coef[2]*x3[i]) + r7.intercept_)

		#sigmoid function:
		predicted[i] = np.exp(predicted[i])/(1+np.exp(predicted[i]))
		mse += (actual[i] - predicted[i])**2

		#tanh function:
		predicted2[i] = np.tanh(predicted[i])
		mseT += (actual[i] - predicted2[i])

	print("non-linear mse: " + str(mse/2067))
	print("non-linear r^2: " + str(R2(test_norm[4], predicted)) + "\n")

	print("non-linear mseT: " + str(mseT/2067))
	print("non-linear r^2: " + str(R2(test_norm[4], predicted2)))

f(test_norm[4])
	


#A scatter plot that shows linear regression
x = train_norm[0] 
y = train_norm[4]
coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef) 
pyplot.plot(x, train_norm[1], train_norm[2], train_norm[3], y, 'yo', x, poly1d_fn(x), '--k')
pyplot.show()
