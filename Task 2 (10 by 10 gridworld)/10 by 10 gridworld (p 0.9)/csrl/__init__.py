"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
import random
import matplotlib
from itertools import product
from .pomdp import GridPOMDP
import os
import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact



class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.
    
    Attributes
    ----------
    shape : (n_pairs, n_qs, n_rows, n_cols, n_actions)
        The shape of the product MDP.
    
    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function of the star-MDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.
        
    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.
    
    Parameters
    ----------
    pomdp : pomdp.GridPOMDP
        The POMDP that models the environment.
        
    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.
        
    discount : float
        The discount factor.
    
    discountB : float
        The discount factor applied to B states.
    
    """
    def __init__(self, pomdp, oa, discount=0.9, discountB=0.99):
        self.pomdp = pomdp
        self.oa = oa
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.shape = oa.shape + pomdp.shape + (len(pomdp.A)+oa.shape[1],)
        self.shape_Q = oa.shape + pomdp.shape + (len(pomdp.A),)
        self.obsv_elem_size = self.shape[2]*self.shape[3]
        self.num_label = 4 # set q = self.shape[1], set label = 4 (empty:0, a:1, b:2, c:3)
        self.current_label=0
        self.label_uncertainty=0.1 # Pl=1-0.1=0.9
        
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=np.object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(pomdp.A))) + [len(pomdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        self.a_array = np.array([[('a',),       ()]],dtype=np.object)
        self.b_array = np.array([[('b',),       ()]],dtype=np.object)
        self.c_array = np.array([[('c',),       ()]],dtype=np.object)
        
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            label_array = pomdp.label[r,c]
            
            if oa.acc[q][label_array][i]:
                self.reward[i,q,r,c] = 10. #-self.discountB  # oa.acc[q][label_array[0],][i] else 0
            
        # Create the transition matrix
        self.transition_probs = np.empty(self.shape,dtype=np.object)  # Enrich the action set with epsilon-actions
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                if action < len(self.pomdp.A): # MDP actions
                    label_array = pomdp.label[r,c]
                    q_ = oa.delta[q][label_array]  # OA transition, [label_array[0],]
                    pomdp_states, probs = pomdp.get_transition_prob((r,c),pomdp.A[action])  # POMDP transition
                    self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in pomdp_states], probs  
                else:  # epsilon-actions
                    self.transition_probs[i,q,r,c][action] = ([(i,action-len(pomdp.A),r,c)], [1.])
        
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_pomdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i,q,r,c in product(range(n_pomdps),range(n_qs),range(n_rows),range(n_cols)):
            yield i,q,r,c
    
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_pomdps, n_qs, n_rows, n_cols, n_actions = self.shape
        pomdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_pomdps),np.random.randint(n_qs)) + pomdp_state
    
    # convert csrl state to state index (tuple,)
    def state_coord_q(self, obsv_state, state):
        q_state = state[1]
        n_rows, n_cols = self.shape[2], self.shape[3]
        
        coord = (q_state*(n_rows*n_cols) + obsv_state[0]*n_cols + obsv_state[1],)
        return coord
    
    # convert obsv state to state index
    def state_coord(self, obsv_state):
        n_rows, n_cols = self.shape[2], self.shape[3]
        coord = (obsv_state[0]*n_cols + obsv_state[1])
        return coord
    
    # convert csrl state to state index (plot path)
    def path_state_coord(self, state):
        q_state = state[1]
        n_rows, n_cols = self.shape[2], self.shape[3]
        coord = q_state*(n_rows*n_cols) + state[2]*n_cols + state[3]
        return coord
    
    # one hot encoding, output encoded [row|col]
    def state_one_hot_encoding(self, state_index):
        output_row = np.zeros(self.shape[2])
        output_col = np.zeros(self.shape[3])
        output_row[state_index[0]]=1.
        output_col[state_index[1]]=1.
        output_state = [output_row, output_col]
        return output_state
    
    # one hot encoding, output encoded [label or q_state]; only record different label, q_state
    def label_q_encoding(self, label_q):
        output_label_q = np.zeros(self.num_label)
        output_label_q[label_q]=1.
        return output_label_q
    
    # convert to label sequence (FIFO)
    def convert_label(self, state):
        label_array = self.pomdp.label[state[2],state[3]]
        label = 0
        if label_array==self.a_array[0][0]:
            label=1
        elif label_array==self.b_array[0][0]:
            label=2
        elif label_array==self.c_array[0][0]:
            label=3
        if label != self.current_label and label != 0:
            self.current_label = label
        return self.current_label
    
    def train_DRQN(self,start=None, EPISODES=None, num_steps=None, batch_size=None, weights_update=None, state_sequence_size=None, label_sequence_size=None):
        from dqn_rnn import DRQNAgent
        EPISODES=EPISODES if EPISODES else 20000
        num_steps=num_steps if num_steps else 300
        batch_size=batch_size if batch_size else 32
        weights_update=weights_update if weights_update else 50
        state_sequence_size=state_sequence_size if state_sequence_size else 1
        label_sequence_size=label_sequence_size if label_sequence_size else 1
        
        # action size
        #prod_action_size = self.shape[4]
        prod_action_size = 4
        # Observation size
        obsv_size = self.shape[2]+self.shape[3] # state size = row + col (one-hot-encoding: 00010|00100)
        # initialize the DQNAgent
        agent = DRQNAgent(obsv_size, prod_action_size, state_sequence_size, label_sequence_size, self.num_label)
        
        ### if want to continue train the model
        #agent.load("./save/file_name.h5")
        
        num_episode_for_reward = 10 # print the accumulated reward per num of episode
        # initialize the list for plot
        accumulated_rewards=[]
        exploration_rate=[]
        average_rewards_hundred_eps = []
        exploration_rate_hundred_steps = []
        # record the starting time
        import time
        start = time.perf_counter() # record the starting time
        
        for e in range(EPISODES):
            accumulated_rewards_per_episode=0
            eps_count = 0
            label_check_pos = 0
            state_sequence = []
            label_sequence = list(np.zeros((label_sequence_size, self.num_label), dtype=int))
            next_state_sequence = []
            next_label_sequence = list(np.zeros((label_sequence_size, self.num_label), dtype=int))
            
            # ------------------------- initialize the 'START state' and avoid the 'traping' and 'block' states --------------------- #
            pomdp_state = self.pomdp.random_state()
            while self.pomdp.label[pomdp_state[0],pomdp_state[1]] == ('c',) or self.pomdp.structure[(pomdp_state[0],pomdp_state[1])]=='B':
                #print('state in c and B, initial state is regenerated')
                pomdp_state = self.pomdp.random_state()
            # convert to the start product state
            state = (self.shape[0]-1,self.oa.q0)+pomdp_state
            print('START state: '+str(state))
            
            # --------------------------- one-hot-encoding the observed (state) & (q state/label) ----------------------------------- #
            obsv_states, obsv_probs = self.pomdp.get_observation_prob(state[-2:])
            obsv_state = self.pomdp.generate_obsv_state(obsv_states, obsv_probs)
            # encodng (state index & q state/label)
            obsv_state_input = self.state_one_hot_encoding(obsv_state)
            obsv_label_input = self.label_q_encoding(state[1]) # select this for Q state seq as input
            #obsv_label_input = self.label_q_encoding(self.convert_label(state)) # select this for label seq as input
            
            # --------------------------- Append the NEXT (state) & (q state/label), as the sequences ------------------------------- #
            next_state_sequence.append(obsv_state_input)
            if len(next_state_sequence) > state_sequence_size:
                next_state_sequence.pop(0)
            if not np.array_equal(np.array(obsv_label_input), next_label_sequence[-1]):
                next_label_sequence.append(obsv_label_input)
                next_label_sequence.pop(0)
            #-------------------------------------------------------------------------------------------------------------------------#
            for step in range(num_steps):
                #print('STATE: '+str(state)+'###########')
                
                # --------------------------- append the (state) & (q state/label), as the sequences -------------------------------- #
                if eps_count == 0:
                    state_sequence.append(obsv_state_input)
                    if len(state_sequence) > state_sequence_size:
                        state_sequence.pop(0)
                    if not np.array_equal(np.array(obsv_label_input), label_sequence[-1]):
                        label_sequence.append(obsv_label_input)
                        label_sequence.pop(0)
                
                # --------------------------- select the action from the approximated q values -------------------------------------- #
                q_values = agent.act(state_sequence, label_sequence)
                q_values = np.reshape(q_values,(prod_action_size,1))
                # generate the epsilon action by possibility
                if len(self.A[state]) > prod_action_size:# exist epsilon moves
                    if random.random() < 0.05: # set as p = 0.05
                        action = random.choice(self.A[state][prod_action_size:len(self.A[state])]) # epsilon move
                        eps_count = 1
                        #print('eps_count=1')
                    else:
                        action = np.argmax(q_values) # POMDP move
                        eps_count = 0
                else:
                    action = np.argmax(q_values) # POMDP move
                    eps_count = 0
                
                # --------------------------- agent moves on POMDP domain ----------------------------------------------------------- #
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                # collect the reward
                action_input = action
                #print('action: '+str(action_input))
                
                # ----------------------------------- PL-POMDP with dynamic event as p=0.9 ------------------------------------------ #
                current_label_check = self.pomdp.label[next_state[2],next_state[3]]
                if label_check_pos == 1 and not self.oa.acc[next_state[1]][current_label_check][next_state[0]]:
                    next_state = (next_state[0],state[1],next_state[2],next_state[3]) # remain the last q state
                    label_check_pos = 0
                    reward = self.reward[next_state]
                elif self.oa.acc[next_state[1]][current_label_check][next_state[0]] and random.random()<self.label_uncertainty:
                    #print('dynamic event occurrenced !')
                    next_state = (next_state[0],state[1],next_state[2],next_state[3]) # remain the last q state
                    label_check_pos = 1
                    reward = 0.
                else:
                    label_check_pos = 0
                    reward = self.reward[next_state]
                #print('reward'+str(reward))
                
                # --------------------------- one-hot-encoding the observed (state) & (q state/label) ------------------------------- #
                # find the observation states' list and the corresponding probabilities
                obsv_states, obsv_probs = self.pomdp.get_observation_prob(next_state[-2:])
                # observe the next state
                obsv_state = self.pomdp.generate_obsv_state(obsv_states, obsv_probs)
                # encodng (state index & q state/label)
                obsv_state_input_next = self.state_one_hot_encoding(obsv_state)
                obsv_label_input_next = self.label_q_encoding(next_state[1]) # select this for Q state seq as input
                #obsv_label_input_next = self.label_q_encoding(self.convert_label(next_state)) # select this for label seq as input
                
                # --------------------------- Append the NEXT (state) & (q state/label), as the sequences --------------------------- #
                if eps_count == 0:
                    next_state_sequence.append(obsv_state_input_next)
                    if len(next_state_sequence) > state_sequence_size:
                        next_state_sequence.pop(0)
                    if not np.array_equal(np.array(obsv_label_input_next), next_label_sequence[-1]):
                        next_label_sequence.append(obsv_label_input_next)
                        next_label_sequence.pop(0)
                
                # --------------------------- the trainning process ----------------------------------------------------------------- #
                if eps_count == 0 and len(state_sequence)==state_sequence_size and len(next_state_sequence)==state_sequence_size:
                    agent.memorize(state_sequence, label_sequence, action_input, reward, next_state_sequence, next_label_sequence)
                
                # --------------------------- assign the next values as the current values ------------------------------------------ #
                state = next_state
                #print('state: '+str(state))
                obsv_state_input = obsv_state_input_next
                obsv_label_input = obsv_label_input_next
                # count the accumulated rewards
                accumulated_rewards_per_episode = accumulated_rewards_per_episode + reward
                
                if step > 50 and state[1] == self.shape[1]-1: # last state: trapping state
                    break
                
                if e > 0 and step > 0 and step % batch_size == 0:
                    agent.replay(EPISODES, e, batch_size)
                
                if e > 0 and step > 0 and step % weights_update == 0:
                    agent.eval_to_tar()
                #---------------------------------------------------------------------------------------------------------------------#
            
            agent.decay(EPISODES, e)
            print("episode: {}/{}, steps: {}, e: {:.2}".format(e, EPISODES, step+1, agent.epsilon))
            print('accumulated_rewards_per_episode: '+str(accumulated_rewards_per_episode))
            accumulated_rewards.append(accumulated_rewards_per_episode)
            exploration_rate.append(agent.epsilon)
            
            if len(accumulated_rewards)>=num_episode_for_reward:
                average_rewards_hundred_eps.append(np.average(accumulated_rewards))
                accumulated_rewards = []
                exploration_rate_hundred_steps.append(np.average(exploration_rate))
                exploration_rate = []   
        agent.save("./save/Q_seq5_label3_task2_p09.h5")

        ######### print out the total time used for computing #########
        finish = time.perf_counter() # record the finish time
        print(f'Finished in {round(finish-start, 4)} second(s)')

        ########## plot the rewards graph ###########
        t2 = np.arange(0, len(average_rewards_hundred_eps)*num_episode_for_reward, num_episode_for_reward)
        
        return t2, exploration_rate_hundred_steps, average_rewards_hundred_eps
    
    def verify_DRQN(self,start=None, EPISODES=None, num_steps=None, state_sequence_size=None, label_sequence_size=None):
        from dqn_rnn import DRQNAgent
        EPISODES=EPISODES if EPISODES else 10
        num_steps=num_steps if num_steps else 30
        state_sequence_size=state_sequence_size if state_sequence_size else 1
        label_sequence_size=label_sequence_size if label_sequence_size else 1
        
        # action size
        #prod_action_size = self.shape[4]
        prod_action_size = 4
        # Observation size
        #obsv_size = self.shape[1]*self.shape[2]*self.shape[3]
        obsv_size = self.shape[2]+self.shape[3]
        # initialize the DQNAgent
        agent = DRQNAgent(obsv_size, prod_action_size, state_sequence_size, label_sequence_size, self.num_label)
        
        ### if want to continue train the model
        agent.load("./save/Q_seq5_label3_task2_p09.h5")
        
        num_episode_for_reward = 10 # print the accumulated reward per num of episode
        Path = np.zeros((EPISODES, num_steps+1))
        
        for e in range(EPISODES):
            accumulated_rewards_per_episode=0
            eps_count = 0
            label_check_pos = 0
            state_sequence = []
            label_sequence = list(np.zeros((label_sequence_size, self.num_label), dtype=int))
            
            # initialize the 'START state' and avoid the 'traping' and 'block' states
            #mdp_state = self.mdp.random_state()
            pomdp_state = (4,7)
            while self.pomdp.label[pomdp_state[0],pomdp_state[1]] == ('c',) or self.pomdp.structure[(pomdp_state[0],pomdp_state[1])]=='B':
                #print('state in c and B, state is regenerated')
                pomdp_state = self.pomdp.random_state()
                #pomdp_state = (random.randint(0,1),0)
            # convert to the start product state
            #pomdp_state = (9, 0)
            state = (self.shape[0]-1,self.oa.q0)+pomdp_state
            print('START state: '+str(state))
            
            Path[e][0] = self.path_state_coord(state)
            
            # find the observation states' list and the corresponding probabilities
            obsv_states, obsv_probs = self.pomdp.get_observation_prob(state[-2:])
            # initialize the observation
            obsv_state = self.pomdp.generate_obsv_state(obsv_states, obsv_probs)
            obsv_state_input = self.state_one_hot_encoding(obsv_state)
            obsv_label_input = self.label_q_encoding(state[1]) # select this for Q state seq as input
            #obsv_label_input = self.label_q_encoding(self.convert_label(state)) # select this for label state seq as input
            
            for step in range(num_steps):
                #print('STATE: '+str(state))
                #print('obsv_input: '+str(obsv_input))
                
                # --------------------------- Append the (state) & (q state/label), as the sequences --------------------------- #
                if eps_count == 0:
                    state_sequence.append(obsv_state_input)
                    if len(state_sequence) > state_sequence_size:
                        state_sequence.pop(0)
                    if not np.array_equal(np.array(obsv_label_input), label_sequence[-1]):
                        label_sequence.append(obsv_label_input)
                        label_sequence.pop(0)
                #----------------------------------------------------------------------------------------------------------------#
                
                # select the action from the approximated q values
                q_values = agent.act_trained(state_sequence, label_sequence)
                #print('q values: '+str(q_values))
                q_values = np.reshape(q_values,(prod_action_size,1))
                
                if len(self.A[state]) > prod_action_size:# exist epsilon moves
                    if random.random() < (len(self.A[state])-prod_action_size)/len(self.A[state]):
                        action = random.choice(self.A[state][prod_action_size:len(self.A[state])]) # epsilon move
                        eps_count = 1
                    else:
                        action = np.argmax(q_values) # POMDP move
                        eps_count = 0
                else:
                    action = np.argmax(q_values) # POMDP move
                    eps_count = 0
                print('action: '+str(action))
                
                ################### The agnet moves on POMDP simualtion ################
                
                # agent moves to the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                
                # ----------------------------------- PL-POMDP with dynamic event as p=0.9 ------------------------------------------ #
                current_label_check = self.pomdp.label[next_state[2],next_state[3]]
                if label_check_pos == 1 and not self.oa.acc[next_state[1]][current_label_check][next_state[0]]:
                    next_state = (next_state[0],state[1],next_state[2],next_state[3]) # remain the last q state
                    label_check_pos = 0
                    reward = self.reward[next_state]
                elif self.oa.acc[next_state[1]][current_label_check][next_state[0]] and random.random()<self.label_uncertainty:
                    print('dynamic event occurrenced !')
                    next_state = (next_state[0],state[1],next_state[2],next_state[3]) # remain the last q state
                    label_check_pos = 1
                    reward = 0.
                else:
                    label_check_pos = 0
                    reward = self.reward[next_state]
                print('reward'+str(reward))
                
                # find the observation states' list and the corresponding probabilities
                obsv_states, obsv_probs = self.pomdp.get_observation_prob(next_state[-2:])
                # observe the next state
                obsv_state = self.pomdp.generate_obsv_state(obsv_states, obsv_probs)
                obsv_state_input_next = self.state_one_hot_encoding(obsv_state)
                obsv_label_input_next = self.label_q_encoding(next_state[1]) # select this for Q state seq as input
                #obsv_label_input_next = self.label_q_encoding(self.convert_label(next_state)) # select this for label state seq as input
                
                # make the next values as the current values
                state = next_state
                obsv_state_input = obsv_state_input_next
                obsv_label_input = obsv_label_input_next
                print('state: '+str(state))
                
                Path[e][step+1] = self.path_state_coord(state)
                
            print("episode: {}/{}, steps: {}, e: {:.2}".format(e, EPISODES, step+1, agent.epsilon))
            
        return Path
    