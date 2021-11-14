import numpy as np
import os


buffer_size = int(1e9)
observations = np.zeros((buffer_size,17))
actions = np.zeros((buffer_size,6))
next_observations = np.zeros((buffer_size,17))
index = 0
prefix = "data/replay buffer"
begin = 0
end = 6
for i in range(begin, end+1):
    count = 0
    try:
        with open(f'{prefix}/run {i}/observations.npy', 'rb') as obs, open(
            f'{prefix}/run {i}/actions.npy', 'rb') as act, open(
            f'{prefix}/run {i}/next_observations.npy', 'rb') as next_obs:
            try:
                while True:
                    temp_obs = np.load(obs)
                    temp_act = np.load(act)
                    temp_nextobs = np.load(next_obs)
                    if i != 6 or count % 100 == 0:
                        size = temp_obs.shape[0]
                        observations[index:size + index] = temp_obs
                        actions[index:size + index] = temp_act
                        next_observations[index:size + index] = temp_nextobs
                        index += size
                    count += 1
            except ValueError:
                print(f"\nend of run {i}, {count} lines\n")
    except FileNotFoundError:
        print(f"run {i} not found")
        if i == 0:
            exit(0)

observations = observations[:index]
actions = actions[:index]
next_observations = next_observations[:index]

with open(f'{prefix}/concat_obs.npy', 'wb') as f:
    np.save(f,observations)
with open(f'{prefix}/concat_acts.npy', 'wb') as f:
    np.save(f,actions)
with open(f'{prefix}/concat_nextobs.npy', 'wb') as f:
    np.save(f,next_observations)