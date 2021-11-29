import matplotlib.pyplot as plt

import numpy as np

import pickle

from pose_graph_prediction.helpers.defaults import MODEL_DIRECTORY


# Open results file and put contents into results list
results = []
with (open(MODEL_DIRECTORY + "results", 'rb')) as openfile:
    while True:
        try:
            results.append(pickle.load(openfile))
        except EOFError:
            break

results = np.array(results)

# Plot loss curves
epoch_ids = np.arange(results.shape[1])

fig, ax = plt.subplots()
_1 = ax.plot(epoch_ids, results[0, :, 1], label='train_loss')
_2 = ax.plot(epoch_ids, results[0, :, 2], label='test_loss')
_7 = ax.plot(epoch_ids, results[0, :, 1] + results[0, :, 2], label='loss_sum')

ax.legend()

ax.set(xlabel='epoch', ylabel='loss',
       title='Losses per epoch')
ax.grid()

# fig.savefig("loss_curves.png")
plt.show()
