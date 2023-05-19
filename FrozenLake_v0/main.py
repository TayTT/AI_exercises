import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")
environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

def train(epochs, alpha, gamma):

    # QTable: rows -> every state s, cols -> every action a, cells -> quality of the action in the state s
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n ))
    qtable_init = copy.deepcopy(qtable)
    outcomes = []

    for i in range(epochs):
        if i % 100 == 0: print(f"Train epoch: {i}")
        state = environment.reset()[0]
        terminated = False


        while not terminated:
            # Kolumna - jakość akcji -> wybierz akcję o najwyższej wartości lub losową jeśli 0
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])
            else:
                action = environment.action_space.sample()

            observation, reward, terminated, truncated, info = environment.step(action)

            qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[int(observation)])
                                                                     - qtable[state, action])

            state = observation

            if reward:
                outcomes.append(1)
            if terminated and reward == 0.0:
                outcomes.append(0)
                break
    return qtable, outcomes


def train_penalized(epochs, alpha, gamma):

    # QTable: rows -> every state s, cols -> every action a, cells -> quality of the action in the state s
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n ))
    qtable_init = copy.deepcopy(qtable)
    outcomes = []
    penalty = 0

    for i in range(epochs):
        if i % 100 == 0: print(f"Penalized train epoch: {i}")
        state = environment.reset()[0]
        terminated = False


        while not terminated:
            # Kolumna - jakość akcji -> wybierz akcję o najwyższej wartości lub losową jeśli 0
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])
            else:
                action = environment.action_space.sample()

            observation, reward, terminated, truncated, info = environment.step(action)

            qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * (np.max(qtable[int(observation)]) - penalty) - qtable[state, action])

            state = observation

            if reward:
                outcomes.append(1)
                penalty = 0

            if terminated and reward == 0.0:
                outcomes.append(0)
                penalty = -1

    return qtable, outcomes


def test(test_epochs):
    test_outcomes = []
    success_evaluation = []

    for i in range(10):
        print(f"Test # {i}")
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
    return success_evaluation


epochs = 10000
learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
discount_factor = 0.9

test_epochs = 1000
evaluation_trials = 10


eval_percentage = np.empty([len(learning_rate), evaluation_trials])
eval_penalized_percentage = np.empty([len(learning_rate), evaluation_trials])

for l in range(len(learning_rate)):
    qtable, outcomes = train(epochs, learning_rate[l], discount_factor)
    success_evaluation = test(test_epochs)
    tmp = [i/test_epochs*100 for i in success_evaluation]
    eval_percentage = np.vstack([eval_percentage, tmp])

for l in range(len(learning_rate)):
    qtable, outcomes = train_penalized(epochs, learning_rate[l], discount_factor)
    success_evaluation = test(test_epochs)
    tmp = [i/test_epochs*100 for i in success_evaluation]
    eval_penalized_percentage = np.vstack([eval_penalized_percentage, tmp])

print(eval_percentage, eval_penalized_percentage)
# eval_percentage_df = pd.DataFrame(eval_percentage,
#                                   columns=["Trial " + str(i) for i in range(10)],
#                                   index=["Learning rate" + str(learning_rate[i]) for i in range(len(learning_rate))])
# eval_penalized_percentage_df = pd.DataFrame(eval_penalized_percentage,
#                                             columns=["Trial " + str(i) for i in range(10)],
#                                             index=["Learning rate" + str(learning_rate[i]) for i in range(len(learning_rate))])

eval_percentage_df = pd.DataFrame(eval_percentage)
eval_penalized_percentage_df = pd.DataFrame(eval_penalized_percentage)

eval_percentage_df.to_csv("./evaluation_data.csv")
eval_penalized_percentage_df.to_csv("./evaluation_penalized_data.csv")



# print('Q-table after training:')
# print(qtable)
#
# plt.plot(outcomes)
# plt.title("Goal reached in every epoch")
# plt.xlabel("Run number")
# plt.ylabel("Outcome")
# plt.show()


# print(f"In last iteration there were {test_outcomes.count(1)} successes. Success rate: {test_outcomes.count(1) / test_epochs * 100}%")
# eval_percentage = [i/test_epochs*100 for i in success_evaluation]
plt.plot(eval_percentage)
plt.title("Percent of correct paths in each evaluation")
plt.xlabel("Evaluation number")
plt.ylabel("Percentage")
plt.show()

plt.plot(eval_penalized_percentage)
plt.title("Percent of correct paths in each evaluation for penalized algorithm")
plt.xlabel("Evaluation number")
plt.ylabel("Percentage")
plt.show()
