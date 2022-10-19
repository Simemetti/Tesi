import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

f = open('data/data_short_cropped_split.json',)
data = json.load(f)
f.close()

#image = np.array(data["images"][0]).reshape(2*5*59,73*5).transpose()
#image = np.array(data["images"][0]).reshape(200,400)
image = np.array(data["images"][0]).reshape(62,400)


plt.imshow(image, interpolation='nearest')
plt.show()