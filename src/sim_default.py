import numpy as np
from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from MeasurementGen import BaseMeasurementGen
from utils import *

class RewardLogicFull(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        Almost full contribution is required
        """
        percs=np.sum([p["value"] for p in self.model.current_state["perception"]])
        thresh=np.random.uniform(percs*0.8,percs) # almost full contrib
        contribs=np.sum([d["contribution"] for d in decisions])
        outcome=success(thresh,np.sum(contribs))
        if outcome==1:
            costs=np.array([d["cost"] for d in decisions])
            ret=-costs+self.benefit
            ret=[{"reward":r} for r in ret]
        else:
            ret=[{"reward":self.damage}]*self.model.N
        return ret

class RewardLogicUniform(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in self.model.current_state["perception"]])
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
            print("unsuccessful")
            ret=-costs+self.damage
        ret=[{"reward":r} for r in ret]
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
        decs=[a.decisions(p) for a,p in zip(self.model.schedule.agents,perceptions)]
        #print(decs)
        self.last_actions=[{"contribution":(p["value"] if d else np.nan),"cost":p["cost"],"agentID":p["agentID"],"contributed":d,"timestep":p["timestep"]} for p,d in zip(perceptions,decs)]
        return self.last_actions

class DecisionLogicSupervisorMandatory(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],"contributed":True,"timestep":a["timestep"]} for a in perceptions]
        return self.last_actions

class DecisionLogicSupervisorProbabilistic(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        ds=[(True if np.random.uniform()<=0.5 else False) for _ in range(len(perceptions))]
        self.last_actions=[{"contribution":(a["value"] if d else np.nan),"cost":a["cost"],"agentID":a["agentID"],"contributed":d,"timestep":a["timestep"]} for a,d in zip(perceptions,ds)]
        return self.last_actions

class MeasurementGenUniform(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=kwargs["n1"]
        self.n2=kwargs["n2"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[max(1,np.random.randint(self.n1,self.n2)) for _ in population]
        # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
        thresh=len(population) #np.random.randint(1,3)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":max(1,np.random.randint(self.n1,self.n2)),"timestep":timestep,"agentID":i,"threshold":thresh} for i,v in enumerate(vals)]
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
        ret=[{"value":v,"cost":0,"timestep":timestep,"agentID":i,"threshold":thresh} for i,v in zip(range(len(population)),vals)]
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
        ret=[{"value":v,"cost":0,"timestep":timestep,"agentID":i,"threshold":thresh} for i,v in zip(range(len(population)),vals)]
        return ret
