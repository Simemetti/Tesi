import json 
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

data_file = open("data/data_cropped_split.json")
data = json.load(data_file)
data_file.close()

data["labels"] = [item for sublist in data["labels"] for item in sublist]

x_train, x_test, y_train, y_test = train_test_split(data["images"], data["labels"], test_size=0.25)

regressor = SVR(C=10, gamma=0.001,kernel='rbf',epsilon=0.01)
regressor.fit(x_train,y_train)

print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))

pickle_file = open("pickles/data_cropped_split_optimized.pickle","wb+")
pickle.dump(regressor,pickle_file)
pickle_file.close()