import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

f = open('data_short.json',)
data = json.load(f)
f.close()

for i in range(0,len(data["images"])):
    data["images"][i] = np.array(data["images"][i]).reshape(590,365).transpose()
    data["images"][i] = data["images"][i][100:300,100:500]
    data["images"][i] = data["images"][i].reshape(80000,1).ravel().tolist()

with open('data_cropped.json', 'w') as json_file:
    json.dump(data, json_file)