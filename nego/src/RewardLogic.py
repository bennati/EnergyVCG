import numpy as np
from src.RewardLogic import BaseRewardLogic
class NegoRewardLogic(BaseRewardLogic):

    def __init__(self,model):
        np.random.seed()
        self.model=model

    def get_rewards(self,decisions):
        """
        Returns a list of dictionaries containing the reward (float) for each agent
        """
        # TODO why is the reward arbitrarily higher for buyers than for sellers?
        return [{"agentID":d["agentID"],"reward":(4 if d["action"]==2 else 2)} for d in decisions]

# class RewardLogicFull(BaseRewardLogic):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.benefit=5
#         self.damage=-10

#     def get_rewards(self,decisions):
#         """
#         Almost full contribution is required
#         """
#         percs=np.sum([p["value"] for p in self.model.current_state["perception"]])
#         thresh=np.random.uniform(percs*0.8,percs) # almost full contrib
#         contribs=np.sum([d["contribution"] for d in decisions])
#         outcome=success_nego(thresh,np.sum(contribs))
#         if outcome==1:
#             costs=np.array([d["cost"] for d in decisions])
#             ret=-costs+self.benefit
#             ret=[{"reward":r} for r in ret]
#         else:
#             ret=[{"reward":self.damage}]*self.model.N
#         return ret

# class RewardLogicUniform(BaseRewardLogic):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.benefit=5
#         self.damage=-10

#     def get_rewards(self,decisions):
#         """
#         The threshold is randomly generated around the average contribution
#         """
#         percs=[p["production"] for p in self.model.current_state["perception"]]
#         thresh=np.random.normal(loc=np.mean(percs),scale=1)
#         thresh=max(1,thresh)
#         contribs=np.sum([d["contribution"] for d in decisions])
#         outcome=success_nego(thresh,np.sum(contribs))
#         if outcome==1:
#             costs=np.array([d["cost"] for d in decisions])
#             ret=-costs+self.benefit
#             ret=[{"reward":r} for r in ret]
#         else:
#             ret=[{"reward":self.damage}]*self.model.N
#         return ret
