# Q-NN

Replacing Q-table with deep neural network. There are no optimization methods and strategies.

## TODO

- [ ] Applying CNN

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
> Setting: Namespace(e=0.998, lr=0.001, r=2000, s=100, y=0.95)
(Episode:  1999, Agent:     0, Steps:     9)
Score over time: 0.373
```

![Gs](./images/Gs.png)

```
Final Q-Table driven by NN:
array([[1.565, 1.999, 1.8  , 1.863],
       [1.712, 2.145, 1.865, 1.694],
       [1.648, 1.911, 1.832, 1.733],
       [1.603, 1.94 , 1.855, 1.745],
       [1.607, 1.934, 1.892, 1.662],
       [1.704, 1.887, 1.965, 1.834],
       [1.696, 2.095, 1.91 , 1.916],
       [1.831, 2.272, 1.987, 1.713],
       [1.73 , 1.844, 1.986, 1.657],
       [1.526, 1.997, 1.899, 1.679],
       [1.469, 1.967, 1.815, 1.796],
       [1.38 , 1.748, 1.74 , 1.399],
       [1.695, 2.112, 1.976, 2.207],
       [1.946, 2.276, 2.053, 1.844],
       [1.63 , 1.936, 1.974, 1.777],
       [1.549, 1.714, 1.822, 1.538],
       [1.554, 1.732, 1.833, 1.518],
       [1.562, 1.96 , 1.913, 1.745],
       [1.728, 2.092, 2.029, 2.291],
       [2.037, 2.401, 2.14 , 1.939],
       [1.79 , 1.862, 2.096, 1.748],
       [1.518, 1.849, 1.913, 1.533],
       [1.746, 2.67 , 2.477, 2.539],
       [1.375, 1.927, 1.651, 1.729],
       [1.728, 2.051, 1.99 , 2.282],
       [2.186, 2.545, 2.204, 1.553],
       [2.124, 2.569, 2.227, 1.664],
       [1.819, 2.88 , 2.667, 3.368],
       [1.665, 2.263, 1.992, 2.054],
       [2.012, 2.009, 2.437, 1.261],
       [1.553, 1.944, 1.876, 2.241],
       [1.726, 2.326, 2.327, 2.697],
       [1.396, 2.495, 2.451, 2.859],
       [1.545, 2.585, 2.558, 3.023],
       [3.116, 2.645, 2.679, 2.312],
       [1.54 , 2.065, 2.474, 2.16 ]])
Map:
[['Start    ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Obstacle ', 'Normal   '],
 ['Normal   ', 'Normal   ', 'Obstacle ', 'Obstacle ', 'Goal     ', 'Obstacle '],
 ['Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ', 'Normal   ']]
Q-map:
[['Down     ', 'Down     ', 'Down     ', 'Down     ', 'Down     ', 'Left     '],
 ['Down     ', 'Down     ', 'Left     ', 'Down     ', 'Down     ', 'Down     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Left     ', 'Down     '],
 ['Right    ', 'Down     ', 'Left     ', 'Left     ', 'Down     ', 'Down     '],
 ['Right    ', 'Down     ', 'Down     ', 'Right    ', 'Down     ', 'Left     '],
 ['Right    ', 'Right    ', 'Right    ', 'Right    ', 'Up       ', 'Left     ']]
```
