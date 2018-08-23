import numpy as np
from utils import *
class BaseEvaluationLogic():
    def __init__(self):
        np.random.seed()

    def get_evaluation(self,decisions,rewards,threshold=None):
        """
        Computes the measures for this round

        Args:
        decisions: the list of dictionaries for all agents
        rewards: a list of rewards for all agents
        threshold: the value of the success threshold (for contribution)

        Returns:
        A list of dictionaries containing the evaluation of the population behavior
        """
        contributions,values,costs,privs=zip(*[(d["contributed"],d["contribution"],d["cost"],d["privacy"]) for d in decisions])
        cost_pop=[(0 if np.isnan(i) else i)  for i in costs]
        if threshold is None:
            threshold=max([p["threshold"] for p in decisions])
        rews=[i["reward"] for i in rewards]
        tc=np.nansum(values)
        return [{"timestep": decisions[0]["timestep"],
                 "gini":gini(values),
                 "gini_cost":gini(costs),
                 "cost":cost(costs),
                 "cost_pop":cost(cost_pop),
                 "privacy":sum(privs),
                 "social_welfare":social_welfare(cost_pop,rews),
                 "efficiency":efficiency(threshold,np.nansum(values)),
                 "success":success(threshold,np.nansum(values)),
                 "tot_contrib":tc,
                 "num_contrib":tot_contributions([int(c) for c in contributions])}]
## TODO gini for number of contributions
