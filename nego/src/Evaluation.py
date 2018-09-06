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
        # base with bilateral
        # set_new =[({"agent":a,"partner":a.partner_selection_orderbid()}) for a in self.model.schedule.agents]

        # exp 1 and 3 with mediation
        # set_new = self.model.decision_fct.get_partner()

        # exp 2 and 4 with bid split mediation
        # set_new = self.model.decision_fct.get_partner_bidsplit()
        # compute efficiency
        eff = []
        for a in self.model.schedule.agents:
            if a.current_state["partner"] is not None:
                assert(a.get_decision() is not None)
                state=a.current_state["perception"]
                attr=("production" if a.current_state['type']=="seller"
                      else "consumption")              # buyer
                efficiency=1-(state[attr]/state["old_"+str(attr)]) # one if all needs are satisfied, a fraction otherwise
                eff.append(efficiency)
            else:
                eff.append(0)
        rewards=[i["reward"] for i in rewards]
        return [{"social_welfare":social_welfare_new(rewards),
                 #"social_welfare_high":social_welfare_new(rewards_high),
                 #"social_welfare_low":social_welfare_new(rewards_low),
                 "gini":gini([i["action"] for i in decisions]),
                 "efficiency":efficiency_nego(eff,np.count_nonzero(eff)), # sum efficiencies (each is between 0 and 1) and divide by the number of agents whose efficency is greater than 0
                 # TODO is the formula of efficiency correct?
                 "market_access":success_nego(self.model.N,
                                              sum([1 for i in decisions if (i["partner"] is not None and i["action"] is not None)])), # agents that traded
                 #"market_access_high":success_nego(N_low,tot_high_agents),
                 "wealth_distribution":gini(rewards)}]
                 #"wealth_distribution_high":gini(rewards_high),
                 #"wealth_distribution_low":gini(rewards_low),
                 #"market_access_low":success_nego(N_high,tot_low_agents)}]
