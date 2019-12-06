import xlrd
import numpy as np
import statistics
from scipy.stats import linregress
from sklearn import linear_model
from matplotlib import pyplot

loc = ("power_data.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

#Training data
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

#Test data
test1_col = sheet.col_values(0,7502,7652)
test2_col = sheet.col_values(1,7502,7652)
test3_col = sheet.col_values(2,7502,7652)
test4_col = sheet.col_values(3,7502,7652) 
test5_col = sheet.col_values(4,7502,7652) 

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
	print("Corr. Coeff.: " + str(np.corrcoef(test[4], i)) + "\n")

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
print("Slope: " + str(r1.slope))
print("Intercept: " + str(r1.intercept))

r3 = linregress(train_norm[2], train_norm[4])	
print("Slope: " + str(r1.slope))
print("Intercept: " + str(r1.intercept) + "\n")


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

#Making predictions on test data and calculating mean squared error (MSE)
c4 = r4.coef_	
c5 = r5.coef_	
c6 = r6.coef_	
c7 = r7.coef_	
result = []
x1 = test_norm[0]
x2 = test_norm[1]
x3 = test_norm[2]
comp = test_norm[4]
mse1 = mse2 = mse3 = mse4 = mse5 = mse6 = mse7 = 0 
for i in range(0,150):

	output1 = r1.slope * x1[i] + r1.intercept
	mse1 += (comp[i] - output1)

	output2 = r2.slope * x2[i] + r2.intercept
	mse2 += (comp[i] - output2)

	output3 = r3.slope * x3[i] + r3.intercept
	mse3 += (comp[i] - output3)

	output4 = r4.intercept_ + x1[i]*c4[0] + x2[i]*c4[1]
	mse4 += (comp[i] - output4)

	output5 = r5.intercept_ + x1[i]*c5[0] + x2[i]*c5[1]
	mse5 += (comp[i] - output5)

	output6 = r6.intercept_ + x1[i]*c6[0] + x2[i]*c6[1]
	mse6 += (comp[i] - output6)

	output7 = r7.intercept_ + x1[i]*c7[0] + x2[i]*c7[1] + x3[i]*c7[2]
	mse7 += (comp[i] - output7)

print("MSE1: " + str(mse1/150))
print("MSE2: " + str(mse2/150))
print("MSE3: " + str(mse3/150))
print("MSE4: " + str(mse4/150))
print("MSE5: " + str(mse5/150))
print("MSE6: " + str(mse6/150))
print("MSE7: " + str(mse7/150))

#A scatter plot that shows linear regression
#x = train_norm[0]
#y = train_norm[4]
#
#coef = np.polyfit(x,y,1)
#poly1d_fn = np.poly1d(coef) 
#
#pyplot.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
#pyplot.show()
#Creates a scatter plot of our data.
#pyplot.scatter(cols[0], cols[4])
#pyplot.show()
