# Frozen lake
A simple implementation of a Q-Learning algorithm using gym module and a slippery Frozen Lake environment. The program
trains the algorithm using two reword systems: default (where the reward for reaching the goal is 1) and penalized
(where additionally a penalty of -1 is introduced when agent reaches a hole).

## Parameters
The implementation uses a few parameters which specify the course of learning. The user can observe the algorithm's 
success over a number of *learning_rates*, which determine the speed in which the weights of each choice will be 
updated. The *discount_factor* describes the short-sightedness of
the algorithm while picking a next step (0 favours the immediate reward while 1 favours the 
long-term reward).  
The number of *epochs* specifies how many times the algorithm should attempt to solve the environment, while 
*test_epochs* specifies how may times the algorithm should be tested against the environment. The test is then
repeated over a number of *evaluation_trials*, to provide a bigger data sample. 


```
epochs = 1000
learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
discount_factor = 0.9

test_epochs = 100
evaluation_trials = 10
```

## Usage
To run the project please run the
```commandline
main.py
```

## Output
The program produces two files:  
* *"evaluation_data.csv"*, which contains the outcomes of the evaluation for standard reward system
* *"evaluation_penalized_data"*, which contains the outcomes for the evaluation for the penalized reward system