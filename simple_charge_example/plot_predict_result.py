import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib.colors import ListedColormap
emission_max_value = 100
start_time_max = 144

height = 10
pkl_file = './predict_output/predict_7.pkl'

with open(pkl_file, 'rb') as f:
    result = pickle.load(f)

print(result)
# create a simple 2D array
data = np.array(result['action_history'])
start_soc = result['start_soc']
target_soc = result['target_soc']
end_soc = result['current_soc']
print(data)
data = np.array(result['action_history']*height)
data = data.reshape(int(height), int(len(data)/height))

# plot the data with the colormap

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))

fig.suptitle("start_soc {:.3f} target_soc {:.3f} end_soc {:.3f}".format(np.round(start_soc,1), np.round(target_soc,1),
                                                                        np.round(end_soc,1)))

# fig.suptitle("start_soc %f end_soc % f"%(np.round(start_soc,1), np.round(target_soc,1)))
ax1.get_yaxis().set_visible(False)
im = ax1.imshow(data,cmap='Greens')
cbar = fig.colorbar(im, ax=ax1, shrink=0.6)
x = np.linspace(0, int(start_time_max), int(start_time_max + 1))
pollution = emission_max_value / ((start_time_max / 2) ** 2) * (x - (start_time_max / 2)) ** 2
ax2.plot(x,pollution,label = "emission_curve")
ax2.legend()



# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def my_app(cfg : DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))
# my_app()

plt.show()
