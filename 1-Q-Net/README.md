# TODO

- [ ] Adding Grid Search to find optimal hyperparameters

# Concrete example

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
> Setting: Namespace(e=0.998, lr=0.8, r=2000, s=100, y=0.95)
(Episode:  1999, Agent:     0, Steps:     9)
Score over time: 0.485
```

![Gs](./images/Gs.png)

```
Final Q-Table:
array([[ 0.597,  0.63 ,  0.597,  0.629],
       [ 0.629,  0.663,  0.596,  0.597],
       [ 0.595,  0.629,  0.63 ,  0.569],
       [ 0.564,  0.599,  0.59 ,  0.544],
       [ 0.55 ,  0.572,  0.573,  0.521],
       [ 0.524,  0.541,  0.546,  0.52 ],
       [ 0.601,  0.66 ,  0.621,  0.663],
       [ 0.627,  0.698,  0.623,  0.629],
       [ 0.598,  0.66 ,  0.663,  0.603],
       [ 0.569,  0.622,  0.63 ,  0.57 ],
       [ 0.562,  0.599,  0.599,  0.541],
       [ 0.524,  0.559,  0.569,  0.548],
       [ 0.624,  0.693,  0.67 ,  0.698],
       [ 0.656,  0.735,  0.653,  0.662],
       [ 0.629,  0.698,  0.693,  0.617],
       [ 0.608,  0.664,  0.663,  0.601],
       [ 0.575, -0.05 ,  0.63 ,  0.564],
       [ 0.538,  0.538,  0.599,  0.572],
       [ 0.662,  0.733,  0.694,  0.735],
       [ 0.696,  0.774,  0.7  ,  0.714],
       [ 0.662, -0.184,  0.735,  0.668],
       [ 0.629, -0.05 ,  0.698, -0.049],
       [ 0.597,  1.   ,  0.664,  0.535],
       [ 0.566, -0.054, -0.047,  0.537],
       [ 0.697,  0.774,  0.729,  0.773],
       [ 0.747,  0.814,  0.74 , -0.173],
       [ 0.696,  0.858,  0.771, -0.063],
       [ 0.668,  0.904, -0.186,  1.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.536,  0.914,  1.   , -0.037],
       [ 0.733,  0.77 ,  0.783,  0.814],
       [ 0.779,  0.803,  0.772,  0.858],
       [-0.17 ,  0.845,  0.817,  0.903],
       [-0.05 ,  0.904,  0.861,  0.95 ],
       [ 1.   ,  0.954,  0.895,  0.897],
       [-0.052,  0.901,  0.95 ,  0.901]])
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
Q-map:
[['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Left     ', 'Left     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Left     ', 'Left     '],
 ['Right    ', 'Down     ', 'Down     ', 'Down     ', 'Left     ', 'Left     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Down     ', 'Up       '],
 ['Down     ', 'Down     ', 'Down     ', 'Right    ', 'Up       ', 'Left     '],
 ['Right    ', 'Right    ', 'Right    ', 'Right    ', 'Up       ', 'Left     ']]
```
