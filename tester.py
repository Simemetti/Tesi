import json 
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

data_filename = "data_cropped_split"
pickle_filename = "data_cropped_split_optimized"

data_file = open("data/"+data_filename+".json",)
data = json.load(data_file)
data_file.close()

data["labels"] = [item for sublist in data["labels"] for item in sublist]

x_train, x_test, y_train, y_test = train_test_split(data["images"], data["labels"], test_size=0.25)

pickle_file = open("pickles/"+pickle_filename+".pickle","rb")
regressor = pickle.load(pickle_file)
pickle_file.close()

print(regressor.score(x_test, y_test))