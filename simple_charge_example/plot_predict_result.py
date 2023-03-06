import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from matplotlib.colors import ListedColormap


with open('./predict_output/predict_2.pkl', 'rb') as f:
    result = pickle.load(f)

# create a simple 2D array
data = np.array(result['action_history'])
print(data)
data = data[0:30].reshape(1,30)
# data = np.random.rand(1, 100)
print(data)
# define a colormap
colors = ['#FF0000', '#00FF00', '#0000FF']  # red, green, blue
cmap = ListedColormap(colors)

# plot the data with the colormap
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.show()