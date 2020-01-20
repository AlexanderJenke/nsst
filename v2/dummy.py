import numpy as np
import pickle
import math
import matplotlib.pyplot as plt

with open("output/scores.pkl", 'rb') as f:
    d = pickle.load(f)

fig, ax = plt.subplots(1, 3)

for ti, t in enumerate(d):
    img_np = np.zeros((15, 5))
    my_xticks = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '100', '200', '500', '1000']
    my_yticks = ['10', '25', '50', '100', '1000']
    for i, wct in enumerate(my_xticks):
        my_xticks[i] = f"{my_xticks[i]} ({str(d[t][wct]['n_tokens']).zfill(4)})"
        for j, nstates in enumerate(my_yticks):
            if nstates in d[t][wct]["states"]:
                prob = d[t][wct]["states"][nstates]
                img_np[i, j] = prob

    img = ax[ti].imshow(img_np, cmap=plt.cm.bone)
    ax[ti].title.set_text(t)
    ax[ti].set_xticks(range(5))
    ax[ti].set_xticklabels(my_yticks)
    ax[ti].set_yticks(range(15))
    ax[ti].set_yticklabels(my_xticks)
    plt.colorbar(img, ax=ax[ti])
plt.show()

if __name__ == '__main__':
    pass
