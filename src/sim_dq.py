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

# class MeasurementGenUniformDQ(BaseMeasurementGen):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.n1=kwargs["n1"]
#         self.n2=kwargs["n2"]
#         assert(self.n1>=0)
#         assert(self.n2>self.n1)

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         vals=[np.random.uniform(max(1,self.n1),self.n2) for _ in population]
#         costs=[np.random.uniform(max(1,self.n1),self.n2) for _ in population]
#         # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
#         thresh=len(population) #np.random.randint(1,3)
#         assert(thresh<=sum(vals))
#         ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":thresh} for i,(v,c) in enumerate(zip(vals,costs))]
#         return ret

class DecisionLogicSupervisorDQ(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        decs=[a.get_decision() for a in self.model.schedule.agents]
        idxs=[a.unique_id for a in self.model.schedule.agents]
        tmp1=pd.DataFrame(data={"action":decs,"agentID":idxs})
        tmp=pd.merge(pd.DataFrame(perceptions),tmp1,on=["agentID"])
        # print(decs)
        self.act=[{"contribution":(r[1]["value_raw"] if r[1]["action"] else np.nan),"cost":(r[1]["cost_raw"] if r[1]["action"] else np.nan),"privacy":(1 if r[1]["action"] else 0),"agentID":r[1]["agentID"],"contributed":r[1]["action"],"timestep":r[1]["timestep"],"threshold":r[1]["threshold"]} for r in tmp.iterrows()]
        return self.act

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
        return (self.model.current_state["perception"]["value_raw"],self.model.current_state["perception"]["cost_raw"])

    # def get_current_state_int(self):
    #     return (self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])
