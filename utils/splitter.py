import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

f = open('data.json',)
data = json.load(f)
f.close()

new_data = {"images":[],"labels":[]}

for i in range(0,len(data["images"])):
    data["images"][i] = np.array(data["images"][i]).reshape(590,365).transpose()
    data["images"][i] = data["images"][i][100:300,100:500]
    new_data["images"].append(data["images"][i][0:62,].ravel().tolist())
    new_data["images"].append(data["images"][i][62:124,].ravel().tolist())
    new_data["images"].append(data["images"][i][124:186,].ravel().tolist())
    new_data["labels"].append(data["labels"][i])
    new_data["labels"].append(data["labels"][i])
    new_data["labels"].append(data["labels"][i])

with open('data_cropped_split.json', 'w') as json_file:
    json.dump(new_data, json_file)