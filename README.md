# Model-free_DRL_LSTM_on_POMDP_with_LDGBA

## The code is originally from: 
***
***Bozkurt, A. K., Wang, Y., Zavlanos, M. M., & Pajic, M. (2020, May). Control synthesis from linear temporal logic specifications using model-free reinforcement learning. In 2020 IEEE International Conference on Robotics and Automation (ICRA) (pp. 10349-10355). IEEE.***
***
## Usage:
### Folder 'simple go_to_goal gridworld':

1. File 'LSTM and DNN' is the go_to_goal example using either LSTM or DNN.

-  In 'dqn_rnn.py', the Q networks are subject to change.
```
  def __init__(self, state_size, action_size, state_sequence_size):
    self.eval_model = self._build_model_RNN()
    self.tar_model = self._build_model_RNN()
```
-  or use DNN instead.
```
    self.eval_model = self._build_model_DNN()
    self.tar_model = self._build_model_DNN()
```

2. File 'CNN' is the go_to_goal example using 2D CNN.

3. Plot folder has file to plot the cumulative rewards of LSTM, CNN and DNN.



### Folder 'Task 1 (10 by 10 gridworld)':
1. File '10 by 10 gridworld (p 1.0)' is the gridworld example of **static** event.
### Folder 'Task 2 (10 by 10 gridworld)':
1. File '10 by 10 gridworld (p 0.9)' is the gridworld example of **dynamic** event.
2. File '10 by 10 gridworld (p 1.0)' is the gridworld example of **static** event.

-  In 'csrl/__ init __.py' of both '**Task 1**' and '**Task 2**', the labelling uncertainty is subject to change.
```
  self.label_uncertainty=0.1
```
-  The functions of Q state/label sequence are subject to change:

```
  self.label_q_encoding(next_state[1])        # (Default) select this for Q state seq as input, the agent is aware of the task.
  self.label_q_encoding(self.convert_label(next_state)) # select this for label seq as input, the agent is unaware of the task.
```
3. Plot folder has file to plot all the cumulative rewards of both **static events and dynamic events** when the agent is **either aware of the task or not**.

## Installation:
  - Python 3.5+
  - Tensorflow 2.7.0
