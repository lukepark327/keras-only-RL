# PER (DQN-based)

TBA

## References

TBA

## TODO

- [ ] PER typically uses lower learning rates: Decreasing lr (lr /= 4)
- [ ] What is the bottleneck and overhead?
- [ ] Test: Combining DDQN, Dueling DQN, D3QN
- [ ] Test: Combining NoisyNet

## Concrete example

<!--
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
Score over time: 0.055
```

![Gs](./images/Gs.png)

```
Final Q-Table:
array([[ 0.597,  0.636,  0.597,  0.635],
       [ 0.634,  0.666,  0.597,  0.596],
       [ 0.596,  0.634,  0.634,  0.569],
       [ 0.568,  0.598,  0.596,  0.549],
       [ 0.549,  0.575,  0.569,  0.522],
       [ 0.524,  0.545,  0.55 ,  0.527],
       [ 0.595,  0.666,  0.635,  0.666],
       [ 0.633,  0.693,  0.635,  0.634],
       [ 0.597,  0.665,  0.666,  0.599],
       [ 0.569,  0.632,  0.63 ,  0.576],
       [ 0.552,  0.598,  0.6  ,  0.541],
       [ 0.517,  0.569,  0.573,  0.54 ],
       [ 0.636,  0.695,  0.667,  0.693],
       [ 0.666,  0.73 ,  0.667,  0.67 ],
       [ 0.632,  0.693,  0.694,  0.648],
       [ 0.599,  0.667,  0.666,  0.598],
       [ 0.573, -0.05 ,  0.635,  0.568],
       [ 0.543,  0.537,  0.597,  0.566],
       [ 0.67 ,  0.73 ,  0.696,  0.73 ],
       [ 0.694,  0.774,  0.696,  0.694],
       [ 0.666, -0.183,  0.731,  0.668],
       [ 0.633,  0.125,  0.694, -0.053],
       [ 0.599,  1.001,  0.662,  0.518],
       [ 0.565, -0.052, -0.048,  0.537],
       [ 0.697,  0.773,  0.731,  0.774],
       [ 0.73 ,  0.817,  0.73 , -0.182],
       [ 0.705,  0.859,  0.775, -0.049],
       [ 0.666,  0.903, -0.182,  1.   ],
       [ 0.574,  0.441,  0.675,  0.272],
       [ 0.565,  0.916,  0.974,  0.34 ],
       [ 0.731,  0.774,  0.773,  0.817],
       [ 0.774,  0.818,  0.773,  0.859],
       [-0.183,  0.859,  0.818,  0.902],
       [-0.049,  0.902,  0.859,  0.949],
       [ 1.   ,  0.949,  0.903,  0.901],
       [ 0.151,  0.903,  0.95 ,  0.899]])
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
Q-map:
[['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Down     ', 'Left     '],
 ['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Left     ', 'Left     '],
 ['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Left     ', 'Left     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Down     ', 'Up       '],
 ['Right    ', 'Down     ', 'Down     ', 'Right    ', 'Left     ', 'Left     '],
 ['Right    ', 'Right    ', 'Right    ', 'Right    ', 'Up       ', 'Left     ']]
```
-->
