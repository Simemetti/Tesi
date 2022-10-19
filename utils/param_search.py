import json 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  

f = open('data/data_short_cropped_split.json',)
data = json.load(f)
f.close()

data["labels"] = [item for sublist in data["labels"] for item in sublist]

x_train, x_test, y_train, y_test = train_test_split(data["images"], data["labels"], test_size=0.25)

param_grid = {'epsilon':[0.1,0.5,0.01,1]}

regressor = GridSearchCV(SVR(),param_grid,refit=True,verbose=2).fit(x_train,y_train)

print(regressor.best_estimator_)
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))