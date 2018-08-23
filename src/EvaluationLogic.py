import numpy as np
from src.utils import *
class BaseEvaluationLogic():
    def __init__(self):
        np.random.seed()

    def get_evaluation(self,decisions,rewards,threshold):
        """
        Computes the measures for this round

        Args:
        decisions: the list of dictionaries for all agents
        rewards: a list of rewards for all agents
        threshold: the value of the success threshold (for contribution)

        Returns:
        A list of dictionaries containing the evaluation of the population behavior
        """
        contributions,values,costs=zip(*[(d["contributed"],d["contribution"],d["cost"]) for d in decisions])
        cost_pop=[(0 if np.isnan(i) else i)  for i in costs]
        if threshold is None:
            threshold=max([p["threshold"] for p in decisions])
        rews=[i["reward"] for i in rewards] # not necessarily in order of agentID
        tc=tot_contributions([int(c) for c in contributions])
        return [{"timestep": decisions[0]["timestep"],
                 "gini":gini(values),
                 "gini_cost":gini(costs),
                 "cost":cost(costs),
                 "cost_pop":cost(cost_pop),
                 "social_welfare":social_welfare(cost_pop,rews),
                 "efficiency":efficiency(threshold,tc),
                 "success":success(threshold,tc),
                 "tot_contrib":tc,
                 "num_contrib":tot_contributions([int(c) for c in contributions])}]
