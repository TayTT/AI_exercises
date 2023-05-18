import copy
import gym
import numpy as np
import matplotlib.pyplot as plt

epochs = 10000
learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
discount_factor = [0.3, 0.6, 0.9]

test_epochs = 1000
evaluation_trials = 10

# environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")
environment = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

#TODO: implement penalties for stepping into holes


def train(epochs, alpha, gamma):

    # QTable: rows -> every state s, cols -> every action a, cells -> quality of the action in the state s
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n ))
    qtable_init = copy.deepcopy(qtable)
    outcomes = []

    for i in range(epochs):
        if i % 500 == 0: print(i)
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
    return qtable, outcomes


def test(test_epochs):
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
    return success_evaluation


eval_percentage = np.empty([len(learning_rate), evaluation_trials])
for l in range(len(learning_rate)):
    qtable, outcomes = train(epochs, learning_rate[l], discount_factor[-1])
    success_evaluation = test(test_epochs)
    tmp = [i/test_epochs*100 for i in success_evaluation]
    eval_percentage = np.vstack([eval_percentage, tmp])

print(eval_percentage)

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
