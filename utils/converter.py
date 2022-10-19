import json
import numpy
from scipy.io import loadmat
raw_data = loadmat('dataset_SL.mat')

raw_train = numpy.asarray(raw_data["IM_TEST"]).transpose()
raw_label = numpy.asarray(raw_data["IM_TRAIN"]).transpose()

data = {"images":[],"labels":[]}
data["images"] = numpy.concatenate((raw_train,raw_label)).tolist()
data["labels"] = numpy.concatenate((raw_data["LABEL_TEST"],raw_data["LABEL_TRAIN"])).tolist()

print(data.keys())

with open('data.json', 'w') as json_file:
    json.dump(data, json_file)