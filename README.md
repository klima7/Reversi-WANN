# Reversi-WANN

## Overview

Reinforcement Learning in Reversi game with Weight Agnostic Neural Network.

Based on article [weightagnostic.github.io](https://weightagnostic.github.io/) and my other repository [Reversi-RL](https://github.com/klima7/Reversi-RL).

## Running

### Training
```
python wann_train.py -p p/reversi_5_4.json -n 8
```

### Testing
```
python wann_test.py -p p/reversi_5_4.json -r 1000
```

## Results

_Fitness may be interpreted as accuracy_

### Reversi 5x4
```
[***]   Fitness:          [0.48 0.45 0.46 0.8  0.77 0.75]
[***]   Weight Values:    [-2.  -1.  -0.5  0.5  1.   2. ]
```
![Reversi_5_4](https://github.com/klima7/Reversi-WANN/blob/master/log/reversi_5_4.png)
