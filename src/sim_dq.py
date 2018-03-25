from Supervisor import BaseSupervisor
from DecisionLogic import BaseDecisionLogic
from MeasurementGen import *
from utils import *
from brains import DQlearner
import itertools
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import uuid

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

class DecisionLogicSupervisorDQ(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def __init__(self,model,alpha=0.01,gamma=0.0,training=True):
        super().__init__(model)
        possible_values=list(range(max(1,self.model.measurement_fct.n1),self.model.measurement_fct.n2)) # TODO binarize a continuous range
        possible_costs=list(range(max(1,self.model.measurement_fct.n1),self.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.states=list(itertools.product(*[possible_values,possible_costs]*self.model.N)) # all possible states
        self.actions=list(itertools.product([False,True],repeat=self.model.N))
        self.dqlearner=DQlearner(self.states,range(len(self.actions)),gamma=gamma,alpha=alpha,n_features=len(self.states[0]))
        self.act=self.actions[0]
        # if training:
        #     self.dqlearner.train(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)

    def get_decision(self, perceptions):
        current=self.get_current_state()
        qvals=self.dqlearner.sess.run(self.dqlearner.q_eval, feed_dict={self.dqlearner.s: [current]})[0]
        probs=boltzmann(qvals,0.1)
        self.act=np.random.choice(range(len(self.actions)),p=probs)
        act=self.actions[self.act]
        ret=[{"contribution":(a["value"] if act[a["agentID"]] else np.nan),"cost":(a["cost"] if act[a["agentID"]] else np.nan),"privacy":1,"agentID":a["agentID"],"contributed":act[a["agentID"]],"timestep":a["timestep"],"threshold":a["threshold"]} for a in perceptions]
        return ret

    def feedback(self,perceptions,reward,rew_type="reward"):
        current=self.get_current_state()
        ## count only reward of contributors
        # self.reward=np.nanmean(np.array([r[rew_type] for r in reward])[list(self.act)]) # TODO it could be nan
        ## use all rewards
        self.reward=np.mean([r[rew_type] for r in reward])
        self.dqlearner.learn(current,self.states[0],self.act,self.reward)

    def get_qtable(self):
        print(self.actions)
        return self.dqlearner.get_qtable().assign(idx=0)

    def get_qcount(self):
        return self.dqlearner.get_qcount().assign(idx=0)

    def get_current_state(self):
        perceptions=self.model.current_state["perception"]
        pop=list(enumerate([p["agentID"] for p in perceptions])) # indexes at which each agentID is
        pop=sorted(pop,key=lambda x: x[1]) # sort by agentID
        state=[[perceptions[i]["value"],perceptions[i]["cost"]] for i,_ in pop] # collect in order of agentID, ascending
        state=[i for j in state for i in j] # flatten
        return state

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DecisionLogicDQ(BaseDecisionLogic):
    def __init__(self,model,alpha=0.01,gamma=0.0,training=False):
        super().__init__(model)
        possible_values=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        possible_costs=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.states=list(itertools.product(possible_values,possible_costs)) # all possible states
        self.actions=[0,1]
        self.dqlearner=DQlearner(self.states,self.actions,gamma=gamma,alpha=alpha)
        self.act=1
        if training:
            self.dqlearner.train(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)

    def get_decision(self, perception):
        current=self.get_current_state()
        self.act=self.dqlearner.get_decision(current)
        assert(self.act in self.actions)
        return self.act

    def feedback(self,perceptions,reward,rew_type="reward"):
        assert(reward["agentID"]==self.model.unique_id)
        current=self.get_current_state()
        self.reward=reward[rew_type]
        self.dqlearner.learn(current,self.states[0],self.act,self.reward)

    def get_qtable(self):
        return self.dqlearner.get_qtable()

    def get_qcount(self):
        return self.dqlearner.get_qcount()

    def get_current_state(self):
        return (self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])

    # def get_current_state_int(self):
    #     return (self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])
