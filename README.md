# keras-only-RL

Implementating Reinforcement Learning from A to Z using keras only.

<!--
# Contents

TBA

# Atari Breakout

TBA
-->

# How to run

## For example: D3QN

```bash
sh atari_breakout_run.sh --double=True --dueling=True
```

## Help

```bash
sh atari_breakout_run.sh -h
```
```
usage: main.py [-h] [--e E] [--double D] [--dueling B]

Some hyperparameters

optional arguments:
  -h, --help   show this help message and exit
  --e E        Total episodes
  --double D   Enable Double DQN
  --dueling B  Enable Dueling DQN
```

# To the Rainbow

| Technique | Problem | How to solve it |
| --- | --- | --- |
| DQN | Non-stationary targets makes learning unstable | Fixed Q-targets |
|  | Correlation between samples makes W biased | Replay Memory |
| Double ~ | Maximum estimator raises over-estimation | Using double estimators |
| Dueling ~ | Some state may have inherently low value | *Q(s, a) = V(s) + A(s, a)* |

<!--
### NoisyNet (DQN-based) (WIP)

NoisyNet replaces e-greedy heuristics with noise on weights (NoisyDense) .

* Agent's policy can be used to aid efficient exploration.
* Maybe it works better than the e-greedy method when the problem is hard to solve...

### PER

TBA

### C51 (WIP)

A Distributional Perspective on Reinforcement Learning (C51)

### Multi-Step Learning

TBA
-->
