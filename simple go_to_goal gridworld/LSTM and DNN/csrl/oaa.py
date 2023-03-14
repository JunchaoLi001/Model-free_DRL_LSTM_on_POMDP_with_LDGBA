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
'''        
        
    def __init__(self):
        self.q0 = 0
        self.delta = [{(): 0,
  ('a',): 0,
  ('b',): 0,
  ('c',): 1,
  ('a', 'b'): 0,
  ('a', 'c'): 0,
  ('b', 'c'): 0,
  ('a', 'b', 'c'): 0},
 {(): 1,
  ('a',): 1,
  ('b',): 1,
  ('c',): 1,
  ('a', 'b'): 1,
  ('a', 'c'): 1,
  ('b', 'c'): 1,
  ('a', 'b', 'c'): 1}]
        
        self.acc = [{(): [None],
  ('a',): [None],
  ('b',): [True],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [None],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]}]
        
        self.shape = (1, 2)
        self.eps = [[],[]]



class oaa:
    
    def __init__(self):
        self.q0 = 0
        self.delta = [{(): 0,
  ('a',): 1,
  ('b',): 0,
  ('c',): 2,
  ('a', 'b'): 0,
  ('a', 'c'): 2,
  ('b', 'c'): 2,
  ('a', 'b', 'c'): 2},
 {(): 1,
  ('a',): 1,
  ('b',): 0,
  ('c',): 2,
  ('a', 'b'): 0,
  ('a', 'c'): 2,
  ('b', 'c'): 2,
  ('a', 'b', 'c'): 2},
 {(): 2,
  ('a',): 2,
  ('b',): 2,
  ('c',): 2,
  ('a', 'b'): 2,
  ('a', 'c'): 2,
  ('b', 'c'): 2,
  ('a', 'b', 'c'): 2}]
        
        self.acc = [{(): [None],
  ('a',): [True],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [None],
  ('b',): [True],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [None],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]}]
        
        self.shape = (1, 3)
        self.eps = [[], [], []]


class oaa:
    
    def __init__(self):
        self.q0 = 0
        self.delta = [{(): 0,
  ('a',): 0,
  ('b',): 0,
  ('c',): 3,
  ('a', 'b'): 3,
  ('a', 'c'): 3,
  ('b', 'c'): 3,
  ('a', 'b', 'c'): 3},
 {(): 1,
  ('a',): 1,
  ('b',): 1,
  ('c',): 3,
  ('a', 'b'): 3,
  ('a', 'c'): 3,
  ('b', 'c'): 3,
  ('a', 'b', 'c'): 3},
 {(): 2,
  ('a',): 2,
  ('b',): 2,
  ('c',): 3,
  ('a', 'b'): 3,
  ('a', 'c'): 3,
  ('b', 'c'): 3,
  ('a', 'b', 'c'): 3},
 {(): 3,
  ('a',): 3,
  ('b',): 3,
  ('c',): 3,
  ('a', 'b'): 3,
  ('a', 'c'): 3,
  ('b', 'c'): 3,
  ('a', 'b', 'c'): 3}]
        
        self.acc = [{(): [None],
  ('a',): [None],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [True],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [None],
  ('b',): [True],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]},
 {(): [None],
  ('a',): [None],
  ('b',): [None],
  ('c',): [None],
  ('a', 'b'): [None],
  ('a', 'c'): [None],
  ('b', 'c'): [None],
  ('a', 'b', 'c'): [None]}]
        
        self.shape = (1, 4)
        self.eps = [[], [], [], []]
'''


