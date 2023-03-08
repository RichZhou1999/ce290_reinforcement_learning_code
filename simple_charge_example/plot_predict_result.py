import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from matplotlib.colors import ListedColormap


height = 10
pkl_file = './predict_output/predict_0.pkl'

with open(pkl_file, 'rb') as f:
    result = pickle.load(f)

# create a simple 2D array
data = np.array(result['action_history'])
print(data)
data = np.array(result['action_history']*height)
data = data.reshape(int(height), int(len(data)/height))
# data = data[0:30].reshape(1,30)
# data = np.random.rand(1, 100)
# print(data)
# define a colormap
# colors = ['#FF0000', '#00FF00', '#0000FF']  # red, green, blue
# cmap = ListedColormap(colors)

# plot the data with the colormap
plt.imshow(data, cmap='Greens')
ax = plt.gca()
ax.get_yaxis().set_visible(False)
plt.colorbar()
# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels = [item for item in ax.get_xticklabels()]
# print(labels)


# ax.set_xticklabels(labels)
plt.show()