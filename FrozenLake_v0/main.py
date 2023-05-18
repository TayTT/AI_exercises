import copy
import gym
import numpy as np
import matplotlib.pyplot as plt

# Train ======================================================================================
epochs = 10000
alpha = 0.1            # Learning rate
gamma = 0.9            # Discount factor

# environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")
environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

# QTable: rows -> every state s, cols -> every action a, cells -> quality of the action in the state s
qtable = np.zeros((environment.observation_space.n, environment.action_space.n ))
qtable_init = copy.deepcopy(qtable)
outcomes = []

for i in range(epochs):
    if i % 25 == 0: print(i)
    state = environment.reset()[0]
    terminated = False


    while not terminated:
        # Kolumna - jakość akcji -> wybierz akcję o najwyższej wartości lub losową jeśli 0
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()

        observation, reward, terminated, truncated, info = environment.step(action)

        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[int(observation)]) - qtable[state, action])

        state = observation

        if reward:
            outcomes.append(1)
        if terminated and reward == 0.0:
            outcomes.append(0)
            break

print('Q-table after training:')
print(qtable)

plt.plot(outcomes)
plt.title("Goal reached in every epoch")
plt.xlabel("Run number")
plt.ylabel("Outcome")
plt.show()

# Test ======================================================================================
test_epochs = 1000
test_outcomes = []
success_evaluation = []

for i in range(10):
    for i in range(test_epochs):
        state = environment.reset()[0]
        terminated = False

        while not terminated:
            action = np.argmax(qtable[state])
            observation, reward, terminated, truncated, info = environment.step(action)
            state = observation
            if reward:
                test_outcomes.append(1)
            if terminated and reward == 0.0:
                test_outcomes.append(0)
    success_evaluation.append(test_outcomes.count(1))
    test_outcomes = []
print(success_evaluation)

print(f"In last iteration there were {test_outcomes.count(1)} successes. Success rate: {test_outcomes.count(1) / test_epochs * 100}%")
eval_percentage = [i/test_epochs*100 for i in success_evaluation]
plt.plot(eval_percentage)
plt.title("Percent of correct paths in each evaluation")
plt.xlabel("Evaluation number")
plt.ylabel("Percentage")
plt.show()