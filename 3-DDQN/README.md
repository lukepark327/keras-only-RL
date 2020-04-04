# DDQN

DDQN is the Double Q-learning using two deep neural networks. DDQN solves over-estimation problem in Q-learning with double estimator instead of maximum estimator. In DDQN, selecting action and expectation(evaluation) of action value are divided.

## Concrete example

```
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
```

## Result

```
> Setting: Namespace(e=0.989, lr=0.001, r=200, s=100, y=0.95)
(Episode:   199, Steps:     9)
Score over time: -0.12
```

![Gs](./images/Gs.png)

```
Final Q-Table:
array([[0.727, 0.747, 0.731, 0.747],
       [0.75 , 0.867, 0.733, 0.741],
       [0.728, 0.726, 0.743, 0.733],
       [0.75 , 0.822, 0.761, 0.728],
       [0.727, 0.784, 0.787, 0.654],
       [0.624, 0.74 , 0.739, 0.671],
       [0.712, 0.857, 0.761, 0.869],
       [0.741, 1.   , 0.758, 0.73 ],
       [0.765, 0.848, 0.848, 0.801],
       [0.702, 0.743, 0.78 , 0.696],
       [0.712, 0.751, 0.804, 0.717],
       [0.515, 0.722, 0.749, 0.719],
       [0.808, 0.999, 0.858, 0.998],
       [0.872, 1.108, 0.883, 0.843],
       [0.817, 0.959, 0.948, 0.846],
       [0.802, 0.809, 0.831, 0.755],
       [0.724, 0.642, 0.77 , 0.705],
       [0.693, 0.712, 0.761, 0.68 ],
       [0.924, 1.103, 0.979, 1.109],
       [0.999, 1.221, 1.015, 0.98 ],
       [0.92 , 0.317, 1.101, 0.808],
       [0.778, 0.68 , 0.879, 0.649],
       [0.785, 1.743, 1.031, 0.706],
       [0.72 , 0.651, 0.608, 0.706],
       [0.977, 1.23 , 1.108, 1.22 ],
       [1.117, 1.311, 1.108, 0.312],
       [0.986, 1.452, 1.185, 0.631],
       [0.814, 1.508, 0.312, 1.736],
       [0.757, 0.814, 0.667, 0.579],
       [0.702, 1.506, 1.756, 0.835],
       [1.102, 1.206, 1.259, 1.307],
       [1.231, 1.311, 1.236, 1.445],
       [0.314, 1.448, 1.316, 1.552],
       [0.659, 1.549, 1.457, 1.646],
       [1.757, 1.654, 1.57 , 1.549],
       [1.411, 1.553, 1.669, 1.548]])
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
Q-map:
[['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Left     ', 'Down     '],
 ['Right    ', 'Down     ', 'Down     ', 'Left     ', 'Left     ', 'Left     '],
 ['Down     ', 'Down     ', 'Down     ', 'Left     ', 'Left     ', 'Left     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Down     ', 'Up       '],
 ['Down     ', 'Down     ', 'Down     ', 'Right    ', 'Down     ', 'Left     '],
 ['Right    ', 'Right    ', 'Right    ', 'Right    ', 'Up       ', 'Left     ']]
```
