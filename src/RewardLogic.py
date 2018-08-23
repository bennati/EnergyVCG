import numpy as np
from utils import *

class BaseRewardLogic():

    def __init__(self,model):
        np.random.seed()
        self.model=model

    def get_rewards(self,decisions):
        """
        Returns a list of dictionaries containing the reward (float) for each agent
        """
        return [{"reward":0}]*len(decisions)

class RewardLogicUniform(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=10
        self.damage=-10
        self.last_gini=0

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in decisions])
        contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
        outcome=success(thresh,contribs)
        self.last_gini=gini([d["contributed"] for d in decisions])
        # print("gini for contributions "+str([d["contributed"] for d in decisions])+" is "+str(self.last_gini))
        # self.last_gini=int(renormalize(self.last_gini,[0,1],[0,gini_bin_no])) # binarize
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
        if outcome==1:
            ret=-costs+self.benefit
        else:
            # print("unsuccessful")
            ret=-costs+self.damage
        # ret=costs
        # ret=[renormalize(r,[self.damage-self.model.measurement_fct.n2,self.benefit],[0,1]) for r in ret] # contain values or qvalues will explode
        ret=[{"agentID": d["agentID"],"reward":r,"gini":self.last_gini} for r,d in zip(ret,decisions)]
        return ret

class RewardLogicInequalityAversion(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=10
        self.damage=-10

    def get_rewards(self,decisions,e=0.1):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in decisions])
        contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
        # if thresh<=contribs:
        #     print("success "+str(thresh)+" "+str(contribs))
        # else:
        #     print("insuccess "+str(thresh)+" "+str(contribs))
        outcome=success(thresh,contribs)
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])**(1-e)
        if outcome==1:
            ret=-costs+self.benefit
        else:
            # print("unsuccessful")
            ret=-costs+self.damage
        ret=[{"agentID": d["agentID"],"reward":r} for r,d in zip(ret,decisions)]
        return ret
