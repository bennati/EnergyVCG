from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from MeasurementGen import BaseMeasurementGen
from utils import *
import itertools
import numpy as np
import pandas as pd

class DecisionLogicQlearn(BaseDecisionLogic):
    def __init__(self,model,wl=2,gamma = 0.5,alpha = 0.5,epsilon = 0.7):
        super().__init__(model)
        self.gamma = 0 #gamma TODO
        self.alpha = alpha
        self.epsilon = epsilon
        self.last_actions=0
        self.reward=0
        self.possible_values=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.possible_costs=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.actions=[0,1]
        self.payoffs=[0]*len(self.actions)
        self.window_len=0
        self.window_len_other=0
        self.history=[0]*self.window_len
        self.history_other=[0]*self.window_len_other
        self.states=list(itertools.product(self.possible_values,self.possible_costs, # all possible states
                                           list(itertools.product(range(len(self.actions)),repeat=self.window_len)),
                                           list(itertools.product(range(len(self.actions)),repeat=self.window_len_other)))) # all possible histories
        self.q=np.full([len(self.states),len(self.actions)],0.5)

    def get_current_state(self):
        ret=np.argwhere([True if a==self.model.current_state["perception"]["value"] and b==self.model.current_state["perception"]["cost"] and c==tuple(self.history) and d==tuple(self.history_other) else False for a,b,c,d in self.states])[0][0]
        assert(ret in range(len(self.states)))
        return ret

    def get_qtable(self):
        return pd.DataFrame(self.q.round(3),columns=self.actions,index=self.states)

    def update_q(self,state, next_state, action, reward):
        """
        https://gist.github.com/kastnerkyle/d127197dcfdd8fb888c2
        """
        if reward<0:
            # print("Warning, negative reward: "+str(reward))
            reward=0
        qsa = self.q[state, action]
        new_q = qsa + self.alpha * (reward + self.gamma * self.q[next_state, :].max() - qsa)
        self.q[state, action] = new_q
        # renormalize row to be between 0 and 1
        rn = self.q[state][self.q[state] > 0] / np.sum(self.q[state][self.q[state] > 0])
        self.q[state][self.q[state] > 0] = rn
        try:
            assert(np.sum(self.q[state]).round(5)==1)
        except AssertionError:
            print("Warning "+str(self.q[state])+" "+str(np.sum(self.q[state])))


    def get_decision(self,perceptions):
        if self.epsilon > 0.0001: # simulated annealing
            self.epsilon=self.epsilon/1.1
        rnd=""
        current=self.get_current_state()
        if np.random.random()>=self.epsilon and np.sum(self.q[current]) > 0:
            self.last_actions = np.argmax(self.q[current])
        else: # choose random action
            self.last_actions = np.random.choice(self.actions)
            rnd="random "
        # print("agent "+str(self.model.unique_id)+" is choosing "+rnd+"action "+("coop" if self.last_actions==1 else "defect")+" with value "+str(self.value)+" and cost "+str(self.cost))
        return self.last_actions

    def feedback(self,perceptions,reward):
        current=self.get_current_state()
        # update history
        if len(self.history)>0:
            self.history[:-1] = self.history[1:]; self.history[-1] = self.last_actions
            assert(all([x in self.actions for x in self.history]))
            # find matching next states
            nexts=list(np.argwhere([True if c==tuple(self.history) else False for a,b,c,d in self.states]).flatten())
            assert(nexts)       # not empty
            assert(all([x in range(len(self.states)) for x in nexts]))
        else:
            nexts=[0]           # TODO give a reasonable value
        ## TODO only if info about other agetns are given in perception
        # if len(self.history_other)>0:
        #     other=[a for a in self.model.model.schedule.agents if a.unique_id != self.model.unique_id]
        #     self.history_other[:-1] = self.history_other[1:]; self.history_other[-1] = other[0].decision_fct.last_actions
        #     assert(all([x in self.actions for x in self.history_other]))
        self.reward=reward["reward"]
        self.payoffs[self.last_actions]+=self.reward
        self.update_q(current,nexts,self.last_actions,self.reward)
