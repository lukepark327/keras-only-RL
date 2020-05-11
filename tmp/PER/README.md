# PER (DQN-based)

In normal DQN, experience transitions were uniformly sampled from a replay memory regardless of their significance. PER replays important transitions more frequently.

## References

Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

## TODO

- [ ] PER typically uses lower learning rates: Decreasing lr (lr /= 4)
- [ ] Test: Combining DDQN, Dueling DQN, D3QN
- [ ] Test: Combining NoisyNet

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
> Setting: Namespace(e=0.979, lr=0.001, r=200, s=100, y=0.95)
(Episode:   199, Steps:     9)
Score over time: 0.065
```

![Gs](./images/Gs.png)

```
Final Q-Table:
array([[ 0.603,  0.631,  0.603,  0.624],
       [ 0.627,  0.662,  0.603,  0.601],
       [ 0.623,  0.629,  0.623,  0.577],
       [ 0.575,  0.604,  0.603,  0.55 ],
       [ 0.551,  0.575,  0.576,  0.533],
       [ 0.538,  0.558,  0.55 ,  0.535],
       [ 0.601,  0.67 ,  0.631,  0.661],
       [ 0.623,  0.701,  0.64 ,  0.627],
       [ 0.604,  0.664,  0.66 ,  0.602],
       [ 0.578,  0.63 ,  0.629,  0.572],
       [ 0.551,  0.603,  0.602,  0.554],
       [ 0.536,  0.568,  0.57 ,  0.556],
       [ 0.64 ,  0.704,  0.668,  0.701],
       [ 0.655,  0.734,  0.666,  0.68 ],
       [ 0.631,  0.697,  0.701,  0.632],
       [ 0.602,  0.647,  0.664,  0.601],
       [ 0.575, -0.053,  0.631,  0.565],
       [ 0.555,  0.553,  0.602,  0.568],
       [ 0.689,  0.741,  0.601,  0.736],
       [ 0.698,  0.781,  0.563,  0.686],
       [ 0.664, -0.127,  0.737,  0.642],
       [ 0.631, -0.098,  0.693,  0.016],
       [ 0.594,  0.999,  0.474,  0.466],
       [ 0.572,  0.182,  0.1  ,  0.55 ],
       [ 0.707,  0.779,  0.741,  0.781],
       [ 0.737,  0.811,  0.744, -0.099],
       [ 0.908,  0.862,  0.776,  0.466],
       [ 0.39 ,  0.903, -0.104,  0.991],
       [ 0.41 ,  0.689,  0.605,  0.697],
       [ 0.561,  0.945,  0.913,  0.271],
       [ 0.739,  0.777,  0.779,  0.811],
       [ 0.78 ,  0.811,  0.778,  0.858],
       [-0.123,  0.858,  0.809,  0.902],
       [-0.074,  0.904,  0.857,  0.951],
       [ 0.999,  0.954,  0.92 ,  0.904],
       [ 0.389,  0.915,  0.952,  0.886]])
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
Q-map:
[['Down     ', 'Down     ', 'Down     ', 'Down     ', 'Left     ', 'Down     '],
 ['Down     ', 'Down     ', 'Down     ', 'Down     ', 'Down     ', 'Left     '],
 ['Down     ', 'Down     ', 'Left     ', 'Left     ', 'Left     ', 'Left     '],
 ['Down     ', 'Down     ', 'Left     ', 'Left     ', 'Down     ', 'Up       '],
 ['Right    ', 'Down     ', 'Up       ', 'Right    ', 'Right    ', 'Down     '],
 ['Right    ', 'Right    ', 'Right    ', 'Right    ', 'Up       ', 'Left     ']]
```
