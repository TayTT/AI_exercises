import gym
import numpy as np
import random

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
environment.reset()
environment.render()

#QTable: rows -> every state s, cols -> every action a, cells -> quality of the action in the state s

# qtable = np.zeros((16, 4))

# Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n      # = 4
qtable = np.zeros((nb_states, nb_actions))

# random.choice(["LEFT", "DOWN", "RIGHT", "UP"])
action = environment.action_space.sample()

# 2. Implement this action and move the agent in the desired direction
print(environment.step(action))
new_state, reward, done, info = environment.step(action)

# Display the results (reward and map)
environment.render()
print(f'Reward = {reward}')