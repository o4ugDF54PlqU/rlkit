import numpy as np

observations = np.loadtxt("observations.txt")
print("loaded obs")
actions = np.loadtxt("actions.txt")
print("loaded acts")
next_observations = np.loadtxt("next_observations.txt")
print("loaded nxt_obs")

div = 1000
with open('observations.npy', 'wb') as f:
    for i in range(int(observations.shape[0]/div)):
        np.save(f,observations[div*i:div*(i+1)])
with open('actions.npy', 'wb') as f:
    for i in range(int(actions.shape[0]/div)):
        np.save(f,actions[div*i:div*(i+1)])
with open('next_observations.npy', 'wb') as f:
    for i in range(int(next_observations.shape[0]/div)):
        np.save(f,next_observations[div*i:div*(i+1)])