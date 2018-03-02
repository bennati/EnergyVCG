import numpy as np
from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from MeasurementGen import BaseMeasurementGen
from utils import *

class RewardLogicUniform(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
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
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
        if outcome==1:
            ret=-costs+self.benefit
        else:
            # print("unsuccessful")
            ret=-costs+self.damage
        ret=[{"agentID": d["agentID"],"reward":r} for r,d in zip(ret,decisions)]
        return ret

class DecisionLogicEmpty(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        pass

    def feedback(self,perceptions,reward):
        pass

class DecisionLogicSupervisorEmpty(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        decs=[a.get_decision() for a in self.model.schedule.agents]
        idxs=[a.unique_id for a in self.model.schedule.agents]
        tmp1=pd.DataFrame(data={"action":decs,"agentID":idxs})
        tmp=pd.merge(pd.DataFrame(perceptions),tmp1,on=["agentID"])
        #print(decs)
        self.last_actions=[{"contribution":(r[1]["value"] if r[1]["action"] else np.nan),"cost":(r[1]["cost"] if r[1]["action"] else np.nan),"agentID":r[1]["agentID"],"contributed":r[1]["action"],"timestep":r[1]["timestep"],"threshold":r[1]["threshold"]} for r in tmp.iterrows()]
        return self.last_actions

class DecisionLogicSupervisorMandatory(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],"contributed":True,"timestep":a["timestep"],"threshold":a["threshold"]} for a in perceptions]
        return self.last_actions

class DecisionLogicSupervisorProbabilistic(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        ds=[(True if np.random.uniform()<=0.5 else False) for _ in range(len(perceptions))]
        self.last_actions=[{"contribution":(a["value"] if d else np.nan),"cost":(a["cost"] if d else np.nan),"agentID":i,"contributed":d,"timestep":a["timestep"],"threshold":a["threshold"]} for i,(a,d) in enumerate(zip(perceptions,ds))]
        return self.last_actions

class MeasurementGenUniform(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=kwargs["n1"]
        self.n2=kwargs["n2"]
        assert(self.n1>=0)
        assert(self.n2>self.n1)

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[np.random.randint(max(1,self.n1),self.n2) for _ in population]
        costs=[np.random.randint(max(1,self.n1),self.n2) for _ in population]
        # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
        thresh=len(population) #np.random.randint(1,3)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":thresh} for i,(v,c) in enumerate(zip(vals,costs))]
        return ret

class MeasurementGenNormal(BaseMeasurementGen):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.mu=kwargs["mu"]
        self.s=3

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[max(0.01,np.random.normal(loc=self.mu,scale=self.s)) for _ in population]
        thresh=len(population) #np.random.randint(1,10)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":0,"timestep":timestep,"threshold":thresh} for v in vals]
        return ret

class MeasurementGenBinomial(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=1
        self.sep=kwargs["rich"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[(max(0.01,np.random.normal(loc=self.mu1,scale=self.s1))
               if i>len(population)*self.sep else
               max(0.01,np.random.normal(loc=self.mu2,scale=self.s2))) for i in range(len(population))]
        thresh=len(population) #np.random.randint(1,10)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":0,"timestep":timestep,"threshold":thresh} for v in vals]
        return ret
