#!/usr/bin/env python
# coding: utf-8

# In[1]:

### this has only one automaton state, for POMDP problems.

class oaa:
    
    def __init__(self):
        self.q0 = 0
        self.delta = [{(): 0,
  ('a',): 0,
  ('b',): 0,
  ('a', 'b'): 0}]
        
        self.acc = [{(): [None],
  ('a',): [None],
  ('b',): [True],
  ('a', 'b'): [None]}]
        
        self.shape = (1, 1)
        self.eps = [[]]
