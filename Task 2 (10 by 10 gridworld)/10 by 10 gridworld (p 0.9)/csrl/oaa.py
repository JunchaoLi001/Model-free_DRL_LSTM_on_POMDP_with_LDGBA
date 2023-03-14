#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
