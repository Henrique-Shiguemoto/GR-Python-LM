import pandas 					as pd
import matplotlib.pyplot 		as plt
import numpy 					as np

from sklearn.model_selection 	import cross_val_score
from sklearn.model_selection 	import train_test_split
from sklearn.ensemble 			import RandomForestClassifier
from sklearn.metrics 			import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics 			import accuracy_score

def main():
	P1data = loadDataset('P1')
	P2data = loadDataset('P2')
	P3data = loadDataset('P3')
	P4data = loadDataset('P4')

	# FOLD 1
	trainSet = [P2data, P3data, P4data]
	testSet = P1data
	X_Test = testSet.iloc[:, 1:6]
	Y_Test = testSet.iloc[:, [0]]
	trainGestures = pd.concat(trainSet, ignore_index = True)
	X_Train = trainGestures.iloc[:, 1:6]
	Y_Train = trainGestures.iloc[:, [0]]
	n = 200
	clf2 = RandomForestClassifier(n_estimators=n, random_state=0)
	clf2.fit(X_Train, np.ravel(Y_Train))
	prediction = clf2.predict(X_Test)
	cm1 = confusion_matrix(Y_Test, prediction)

	# FOLD 2
	trainSet = [P1data, P3data, P4data]
	testSet = P2data
	X_Test = testSet.iloc[:, 1:6]
	Y_Test = testSet.iloc[:, [0]]
	trainGestures = pd.concat(trainSet, ignore_index = True)
	X_Train = trainGestures.iloc[:, 1:6]
	Y_Train = trainGestures.iloc[:, [0]]
	n = 200
	clf2 = RandomForestClassifier(n_estimators=n, random_state=0)
	clf2.fit(X_Train, np.ravel(Y_Train))
	prediction = clf2.predict(X_Test)
	cm2 = confusion_matrix(Y_Test, prediction)

	# FOLD 3
	trainSet = [P1data, P2data, P4data]
	testSet = P3data
	X_Test = testSet.iloc[:, 1:6]
	Y_Test = testSet.iloc[:, [0]]
	trainGestures = pd.concat(trainSet, ignore_index = True)
	X_Train = trainGestures.iloc[:, 1:6]
	Y_Train = trainGestures.iloc[:, [0]]
	n = 200
	clf2 = RandomForestClassifier(n_estimators=n, random_state=0)
	clf2.fit(X_Train, np.ravel(Y_Train))
	prediction = clf2.predict(X_Test)
	cm3 = confusion_matrix(Y_Test, prediction)

	# FOLD 4
	trainSet = [P1data, P2data, P3data]
	testSet = P4data
	X_Test = testSet.iloc[:, 1:6]
	Y_Test = testSet.iloc[:, [0]]
	trainGestures = pd.concat(trainSet, ignore_index = True)
	X_Train = trainGestures.iloc[:, 1:6]
	Y_Train = trainGestures.iloc[:, [0]]
	n = 200
	clf2 = RandomForestClassifier(n_estimators=n, random_state=0)
	clf2.fit(X_Train, np.ravel(Y_Train))
	prediction = clf2.predict(X_Test)
	cm4 = confusion_matrix(Y_Test, prediction)

	cmFinal = np.zeros((10, 10), dtype=np.int32)
	# sum elements of each cm
	for i in range(0,10):
		for j in range(0,10):
			cmFinal[i][j] = cm1[i][j] + cm2[i][j] + cm3[i][j] + cm4[i][j]

	disp = ConfusionMatrixDisplay(cmFinal, display_labels=['1','2','3','4','5','6','7','8','9','10'])
	disp.plot()
	plt.xlabel("Predicted Gestures")
	plt.ylabel("True Gestures")
	plt.show()

def loadDataset(person):
		if person == 'P1':
			data0 = pd.read_csv("../db/P1/gesto0.csv")
			data1 = pd.read_csv("../db/P1/gesto1.csv")
			data2 = pd.read_csv("../db/P1/gesto2.csv")
			data3 = pd.read_csv("../db/P1/gesto3.csv")
			data4 = pd.read_csv("../db/P1/gesto4.csv")
			data5 = pd.read_csv("../db/P1/gesto5.csv")
			data6 = pd.read_csv("../db/P1/gesto6.csv")
			data7 = pd.read_csv("../db/P1/gesto7.csv")
			data8 = pd.read_csv("../db/P1/gesto8.csv")
			data9 = pd.read_csv("../db/P1/gesto9.csv")
			currentDataFrame = pd.concat([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index = True)
			return currentDataFrame
		elif person == 'P2':
			data0 = pd.read_csv("../db/P2/gesto0.csv")
			data1 = pd.read_csv("../db/P2/gesto1.csv")
			data2 = pd.read_csv("../db/P2/gesto2.csv")
			data3 = pd.read_csv("../db/P2/gesto3.csv")
			data4 = pd.read_csv("../db/P2/gesto4.csv")
			data5 = pd.read_csv("../db/P2/gesto5.csv")
			data6 = pd.read_csv("../db/P2/gesto6.csv")
			data7 = pd.read_csv("../db/P2/gesto7.csv")
			data8 = pd.read_csv("../db/P2/gesto8.csv")
			data9 = pd.read_csv("../db/P2/gesto9.csv")
			currentDataFrame = pd.concat([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index = True)
			return currentDataFrame
		elif person == 'P3':
			data0 = pd.read_csv("../db/P3/gesto0.csv")
			data1 = pd.read_csv("../db/P3/gesto1.csv")
			data2 = pd.read_csv("../db/P3/gesto2.csv")
			data3 = pd.read_csv("../db/P3/gesto3.csv")
			data4 = pd.read_csv("../db/P3/gesto4.csv")
			data5 = pd.read_csv("../db/P3/gesto5.csv")
			data6 = pd.read_csv("../db/P3/gesto6.csv")
			data7 = pd.read_csv("../db/P3/gesto7.csv")
			data8 = pd.read_csv("../db/P3/gesto8.csv")
			data9 = pd.read_csv("../db/P3/gesto9.csv")
			currentDataFrame = pd.concat([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index = True)
			return currentDataFrame
		elif person == 'P4':
			data0 = pd.read_csv("../db/P4/gesto0.csv")
			data1 = pd.read_csv("../db/P4/gesto1.csv")
			data2 = pd.read_csv("../db/P4/gesto2.csv")
			data3 = pd.read_csv("../db/P4/gesto3.csv")
			data4 = pd.read_csv("../db/P4/gesto4.csv")
			data5 = pd.read_csv("../db/P4/gesto5.csv")
			data6 = pd.read_csv("../db/P4/gesto6.csv")
			data7 = pd.read_csv("../db/P4/gesto7.csv")
			data8 = pd.read_csv("../db/P4/gesto8.csv")
			data9 = pd.read_csv("../db/P4/gesto9.csv")
			currentDataFrame = pd.concat([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index = True)
			return currentDataFrame

if __name__ == '__main__':
	main()