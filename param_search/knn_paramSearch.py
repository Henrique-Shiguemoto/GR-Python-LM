import pandas 					as pd
import matplotlib.pyplot 		as plt
import numpy 					as np

from sklearn.model_selection 	import cross_val_score
from sklearn.neighbors 			import KNeighborsClassifier

def main():
	P1data = loadDataset('P1')
	P2data = loadDataset('P2')
	P3data = loadDataset('P3')
	P4data = loadDataset('P4')

	trainSet = [P1data, P2data, P3data, P4data]
	trainGestures = pd.concat(trainSet, ignore_index = True)

	X_Train = trainGestures.iloc[:, 1:6]
	Y_Train = trainGestures.iloc[:, [0]]

	# KNN
	for i in range(1,11):
		classificadorKNN = KNeighborsClassifier(n_neighbors=i)
		scores = cross_val_score(classificadorKNN, X_Train, np.ravel(Y_Train), cv=4, scoring="accuracy")
		print("------------------------ K-FOLD", 4, "-", i, "Neighbors ------------------------")
		print("Scores            :", scores)
		print("Mean Accuracy     :", np.mean(scores))
		print("Standard Deviation:", np.std(scores))
		print("CV                :", np.std(scores) / np.mean(scores))

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