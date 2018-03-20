from DecisionLogic import BaseDecisionLogic
from utils import *
from brains import *
import itertools
import numpy as np

# gini_bin_no=10

# class RewardLogicUniformW1(BaseRewardLogic):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.benefit=10
#         self.damage=-10
#         self.last_gini=0

#     def get_rewards(self,decisions):
#         """
#         The threshold is randomly generated around the average contribution
#         """
#         thresh=max([p["threshold"] for p in decisions])
#         contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
#         outcome=success(thresh,contribs)
#         self.last_gini=gini([d["contribution"] for d in decisions])
#         # print("gini for contributions "+str([d["contributed"] for d in decisions])+" is "+str(self.last_gini))
#         # self.last_gini=int(renormalize(self.last_gini,[0,1],[0,gini_bin_no])) # binarize
#         costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
#         if outcome==1:
#             ret=-costs+self.benefit
#         else:
#             # print("unsuccessful")
#             ret=-costs+self.damage
#         # ret=costs
#         ret=[renormalize(r,[self.damage-self.model.measurement_fct.n2,self.benefit-self.model.measurement_fct.n1],[0,1]) for r in ret] # contain values or qvalues will explode
#         ret=[{"agentID": d["agentID"],"reward":r,"gini":self.last_gini} for r,d in zip(ret,decisions)]
#         return ret

# def renormalize(n, range1, range2):
#     delta1 = range1[1] - range1[0]
#     delta2 = range2[1] - range2[0]
#     return (delta2 * (n - range1[0]) / delta1) + range2[0]

class DecisionLogicWlearn(BaseDecisionLogic):
    def __init__(self,model,gamma = 0.0,alpha = 0.5,tmax=5,training=False):
        super().__init__(model)
        possible_values=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        possible_costs=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.states=list(itertools.product(possible_values,possible_costs)) # all possible states
        self.actions=[0,1]
        self.qlearner=Qlearner(self.states,self.actions,gamma=gamma, alpha=alpha,tmax=tmax)
        if training:
            self.qlearner.train(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)   # pretraining
        self.wlearner=Wlearner(self.states,gamma=gamma, alpha=alpha)
        self.qlearner_gini=Qlearner(self.states,self.actions,gamma=gamma, alpha=alpha,tmax=tmax)
        self.wlearner_gini=Wlearner(self.states,gamma=gamma, alpha=alpha)
        self.act=1

    def get_qtable(self):
        # print("------------------------------")
        # asd=self.qlearner.get_qtable()
        # asd.rename(columns={0:"no",1:"yes"},inplace=True)
        # asd1=self.wlearner.get_qtable()
        # asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        # asd1=self.qlearner_gini.get_qtable()
        # asd1.rename(columns={0:"no_gini",1:"yes_gini"},inplace=True)
        # asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        # asd1=self.wlearner_gini.get_qtable()
        # asd1.rename(columns={"w":"gini"},inplace=True)
        # asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        # print(asd)
        # ret=asd.apply(lambda x: pd.Series(data={1:(x["yes"] if x["w"]>x["gini"] else x["yes_gini"]),0:(x["no"] if x["w"]>x["gini"] else x["no_gini"])}),axis=1)
        # return ret
        return self.qlearner.get_qtable()

    def get_qcount(self):
        return self.qlearner.get_qcount()

    def get_current_state(self):
        ret=(self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])
        assert(ret in self.states)
        return ret

    def get_decision(self,perceptions):
        # print("--------------------")
        current=self.get_current_state()
        # print("evaluating state "+str(current))
        # self.last_gini=self.model.model.reward_fct.last_gini
        policies=[[q.get_decision(current),w.get_decision(current)] for q,w in [[self.qlearner,self.wlearner],[self.qlearner_gini,self.wlearner_gini]]]
        # print(policies)
        # choices,ws=[i.get_decision(current) for i in [self.qlearner,self.wlearner]]
        # choices_gini,ws_gini=[i.get_decision(self.last_gini) for i in [self.qlearner_gini,self.wlearner_gini]]
        # actions=[choices,choices_gini]
        # print("selected actions "+str(actions))
        self.winning_policy=np.argmax([action for action,w in policies])
        # self.winning_policy=0 #np.argmax([ws,ws_gini]) #TODO
        self.act=policies[self.winning_policy][0]
        # print("Ws are "+str([ws,ws_gini])+": winning policy is "+str(self.winning_policy))
        assert(self.act in self.actions)
        return self.act

    def feedback(self,perceptions,reward):
        assert(reward["agentID"]==self.model.unique_id)
        current=self.get_current_state()
        self.reward=reward["reward"]
        next_gini=reward["gini"]
        gini_reward=1-next_gini
        # print("Giving reward "+str(self.reward)+" and gini "+str(gini_reward)+" for state "+str(current))
        self.qlearner.learn(current,self.states[0],self.act,self.reward)
        self.qlearner_gini.learn(current,self.states[0],self.act,gini_reward)
        # update wlearning, only policy that lost
        # print("updating policy "+str(self.winning_policy)+" with rewards "+str(self.reward)+" and "+str(gini_reward))
        if self.winning_policy: # gini won
            self.wlearner.learn(current,self.states[0],self.reward,self.qlearner.get_qvalues(current).max(),0)
        else:
            self.wlearner_gini.learn(current,self.states[0],gini_reward,self.qlearner.get_qvalues(current).max(),0)
