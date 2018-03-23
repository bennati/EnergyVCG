import numpy as np
from DecisionLogic import BaseDecisionLogic
from utils import *

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
        self.act=[{"contribution":(r[1]["value"] if r[1]["action"] else np.nan),"cost":(r[1]["cost"] if r[1]["action"] else np.nan),"privacy":(1 if r[1]["action"] else 0),"agentID":r[1]["agentID"],"contributed":r[1]["action"],"timestep":r[1]["timestep"],"threshold":r[1]["threshold"]} for r in tmp.iterrows()]
        del tmp,decs,idxs,tmp1
        return self.act

class DecisionLogicSupervisorMandatory(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.act=[{"contribution":a["value"],"cost":a["cost"],"privacy":1,"agentID":a["agentID"],"contributed":True,"timestep":a["timestep"],"threshold":a["threshold"]} for a in perceptions]
        return self.act

class DecisionLogicSupervisorDefect(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.act=[{"contribution":0,"cost":0,"privacy":0,"agentID":a["agentID"],"contributed":False,"timestep":a["timestep"],"threshold":a["threshold"]} for a in perceptions]
        return self.act

class DecisionLogicSupervisorProbabilistic(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        ds=[(True if np.random.uniform()<=0.5 else False) for _ in range(len(perceptions))]
        self.act=[{"contribution":(a["value"] if d else np.nan),"cost":(a["cost"] if d else np.nan),"agentID":i,"contributed":d,"timestep":a["timestep"],"threshold":a["threshold"]} for i,(a,d) in enumerate(zip(perceptions,ds))]
        return self.act
