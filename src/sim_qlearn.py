from DecisionLogic import BaseDecisionLogic
from utils import *
from brains import Qlearner
import itertools
import numpy as np
import math
import pandas as pd

class DecisionLogicQlearn(BaseDecisionLogic):
    def __init__(self,model,gamma = 0.5,alpha = 0.5,tmax=5,training=False):
        super().__init__(model)
        possible_values=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        possible_costs=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.states=list(itertools.product(possible_values,possible_costs)) # all possible states
        self.actions=[0,1]
        self.qlearner=Qlearner(self.states,self.actions,gamma=gamma, alpha=alpha,tmax=tmax)
        self.act=1
        if training:
            self.qlearner.train()   # pretraining

    def get_current_state(self):
        ret=(self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])
        assert(ret in self.states)
        return ret

    def get_qtable(self):
        return self.qlearner.get_qtable()

    def get_qcount(self):
        return self.qlearner.get_qcount()

    def get_decision(self,perceptions):
        current=self.get_current_state()
        self.act=self.qlearner.get_decision(current)
        return self.act

    def feedback(self,perceptions,reward):
        assert(reward["agentID"]==self.model.unique_id)
        current=self.get_current_state()
        self.reward=reward["reward"]
        self.qlearner.learn(current,self.states[0],self.act,self.reward)
