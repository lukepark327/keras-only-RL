# studying-RL

Studying and implementating Reinforcement Learning from A to Z without using OpenAI Gym.

* OpenAI Gym style code, but no external library (no OpenAI Gym import)
* Keras-based implementation

# Contents

TBA

# To the Rainbow

### Q-Table

Simple RL using table as a Q-function.

### Q-NN

Replacing table with function approximation using Neural Network (NN). No optimization technique implemented.

### DQN

Deep Q-Network.

* Fixed Q-target
* Replay Memory

### DDQN

Double DQN.

* Using Double Estimator instead of Maximum Estimator.

### D3QN

Dueling Double DQN (D3QN) is the Dueling DQN with a DDQN(Double DQN) method.

* Value Function
* Advantage Function

### NoisyNet (DQN-based)

NoisyNet replaces e-greedy heuristics with noise on weights (NoisyDense) .

* Agent's policy can be used to aid efficient exploration.
* Maybe it works better than the e-greedy method when the problem is hard to solve...

### PER

TBA

### C51

TBA

<!--
A Distributional Perspective on Reinforcement Learning (C51)
-->

### Rainbow

TBA

# How to run

## EX) Q-Table

```bash
sh Q-Table/run.sh
```

## Help

```bash
sh Q-Table/run.sh -h
```
```bash
usage: main.py [-h] [--lr L] [--y Y] [--e E] [--r R] [--s S]

Some hyperparameters

optional arguments:
  -h, --help  show this help message and exit
  --lr L      learning rate
  --y Y       discount factor
  --e E       e-greedy factor
  --r R       total episodes (rounds)
  --s S       total steps per episode
```

# TODO

- [ ] Using jupyter notebook.
- [ ] More descriptions and references.
- [ ] Multi-Step Learning.
- [ ] Moving TODO list into issues.
- [ ] Combining DQN, DDQN, D3QN, NoisyNet, PER into a one file.
- [ ] Adding an action: Stop
- [ ] Evaluating elapsed time
- [ ] GPU
