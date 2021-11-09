import numpy as np
import os


buffer_size = int(1e9)
observations = [[0.]*17]*buffer_size
actions = [[0.]*6]*buffer_size
next_observations = [[0.]*17]*buffer_size
index = 0
prefix = "data/replay buffer"
for i in range(5):
    count = 0
    with open(f'{prefix}/run {i+1}/observations.npy', 'rb') as obs, open(
        f'{prefix}/run {i+1}/actions.npy', 'rb') as act, open(
        f'{prefix}/run {i+1}/next_observations.npy', 'rb') as next_obs:
        try:
            while True:
                temp = np.load(obs)
                size = temp.shape[0]
                observations[index:size + index] = temp.tolist()
                actions[index:size + index] = np.load(act).tolist()
                next_observations[index:size + index] = np.load(next_obs).tolist()
                count += 1
                index += size
        except ValueError:
            print(f"\nend of file, {count} lines\n")

observations = observations[:index]
actions = actions[:index]
next_observations = next_observations[:index]

with open(f'{prefix}/concat_obs.npy', 'wb') as f:
    np.save(f,observations)
with open(f'{prefix}/concat_acts.npy', 'wb') as f:
    np.save(f,actions)
with open(f'{prefix}/concat_nextobs.npy', 'wb') as f:
    np.save(f,next_observations)