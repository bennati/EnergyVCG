from nego.src.utilsnego import *
from src.EvaluationLogic import BaseEvaluationLogic
from src.utils import gini

class NegoEvaluationLogic(BaseEvaluationLogic):
    def __init__(self,model):
        np.random.seed()
        self.model = model

    def get_evaluation(self,decisions,rewards,threshold):
        """
        Computes the measures for this round

        Args:
        decisions: the list of actions for all agents
        rewards: a list of rewards for all agents
        threshold: the value of the success threshold (for contribution)

        Returns:
        A list of dictionaries containing the evaluation of the population behavior
        """
        rewards=[i["reward"] for i in rewards]
        return [{"social_welfare":social_welfare_rawls(rewards),
                 #"social_welfare_high":social_welfare_new(rewards_high),
                 #"social_welfare_low":social_welfare_new(rewards_low),
                 "gini":gini([i["action"] for i in decisions if i is not None]),
                 "efficiency":efficiency_nego(self.model.schedule.agents), # sum efficiencies (each is between 0 and 1) and divide by the number of agents whose efficency is greater than 0
                 # TODO is the formula of efficiency correct?
                 "market_access":market_access(self.model.N,decisions), # agents that traded
                 #"market_access_high":success_nego(N_low,tot_high_agents),
                 "wealth_distribution":gini(rewards)}]
                 #"wealth_distribution_high":gini(rewards_high),
                 #"wealth_distribution_low":gini(rewards_low),
                 #"market_access_low":success_nego(N_high,tot_low_agents)}]
