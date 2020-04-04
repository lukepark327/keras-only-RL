# studying-RL

Studying and implementating Reinforcement Learning from A to Z without using OpenAI Gym.

* OpenAI Gym style code, but no external library (no OpenAI Gym import)

# Contents

## Value Based Methods

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

### PER

TBA

### NoisyNet

TBA

### C51

TBA

### Rainbow

TBA

# How to run

```bash
sh 0-Q-Table/run.sh
```

# TODO

- [ ] Using jupyter notebook.
- [ ] More descriptions and references.
- [ ] Removing unnecessary code duplications like `env.py`, `visualization.py` and `arguments.py`.
- [ ] Referencing papers.
