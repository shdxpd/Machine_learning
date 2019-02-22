import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data,predict,k=3):
	if(len(data) >= k):
		warnings.warn('K is set to a value less than total voting groups')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append(([euclidean_distance,group]))
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/k
	return vote_result, confidence

accuracies = []
for i in range(20):
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.replace('?',-99999,inplace=True)
	df.drop(['id'],1,inplace=True)

	full_data = df.astype(float).values.tolist()
	random.shuffle(full_data)

	test_size = 0.2
	train_set = {2:[],4:[]}
	test_set = {2:[],4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote,confidences = k_nearest_neighbors(train_set,data,k=20)
			if group == vote:
				correct += 1
			else:
				print(confidences)
			total += 1
	print('Accuracy:',correct/total)
	accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))

# import numpy as np
# from sklearn import preprocessing,neighbors
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# df.replace('?',-99999,inplace=True)
# df.drop(['id'],1,inplace=True)
#
# x = np.array(df.drop(['class'],1))
# y = np.array(df['class'])
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
#
# clf = neighbors.KNeighborsClassifier()
# clf.fit(x_train,y_train)
#
# accuracy = clf.score(x_test,y_test)
# print(accuracy)
#
# example_measures = np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]])
# # print(example_measures)
# example_measures = example_measures.reshape(len(example_measures),-1)
# # print(example_measures)
#
# prediction = clf.predict(example_measures)
# print(prediction)