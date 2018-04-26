import numpy as np
from copy import copy
from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from utils import success

class DecisionLogicAspiration(BaseDecisionLogic):
    """
    Aspiration learning
    """
    def __init__(self,model,**kvargs):
        super().__init__(model)
        self.act=0
        self.aspir_lvl=0.0
        self.actions=[0,1]
        self.payoffs=np.zeros(len(self.actions))
        self.l=10e-4
        self.c=0.05
        self.h=0.01
        self.eta=0.05
        self.eps=10e-4
        self.upperbound=1.1     # < than min utility
        self.lowerbound=-1.1 #-0.5   # > than max utility
        self.delta=0
        assert(self.eta >0)
        assert(self.eps >0)
        assert(self.l >=0)
        assert(self.c >0)
        assert(self.h >0 and self.h <1)
        self.r=lambda: (self.eta*2*np.random.random_sample()-self.eta) if np.random.random()<self.l else 0
        self.sat=lambda x: min(self.upperbound,max(self.lowerbound,x))
        self.phi=lambda z: max(self.h,1+self.c*z) if z<0 else 1

    # def wheel_sel(choices,weights):
    #     w=copy(weights)
    #     w=np.asarray(w)-min(w)
    #     thresh=np.random.uniform(0,max(np.asarray(w)[choices]))
    #     np.random.shuffle(choices)
    #     cumulative=0
    #     for a in choices:
    #         cumulative+=w[a]
    #         if cumulative> thresh:
    #             return a
    #     return 0            # the first and only choice had weight 0

    def get_decision(self,perceptions):
        # print("agent "+str(self.model.unique_id)+" updates action with prob "+str(self.phi(self.delta)))
        if np.random.random()<1-self.phi(self.delta):
            # choose a random action other than the selected one
            rand_act=copy(self.actions)
            rand_act.remove(self.act)
            rand_act=np.random.choice(rand_act)
            # rand_act=wheel_sel(rand_act,self.payoffs)
            self.act=rand_act
            #print("agent "+str(self.model.unique_id)+" is updating action: "+str(self.act))
        return self.act

    def feedback(self,perceptions,reward):
        reward=reward["reward"]
        # update action
        self.payoffs[self.actions.index(self.act)]+=reward
        self.delta=reward-self.aspir_lvl
        # update aspiration level
        self.aspir_lvl=self.sat(self.aspir_lvl+self.eps*self.delta+self.r())
