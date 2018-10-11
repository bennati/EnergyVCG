from src.Supervisor import BaseSupervisor
from nego.src.Agent import NegoAgent
from nego.src.MeasurementGen import MeasurementGenReal
from nego.src.DecisionLogic import NegoDecisionLogic
from src.DecisionLogic import BaseDecisionLogic
from nego.src.RewardLogic import NegoRewardLogic
from nego.src.Evaluation import NegoEvaluationLogic
from mesa import Model
import operator

class NegoSupervisor(BaseSupervisor):

    def __init__(self,N,measurement_fct=MeasurementGenReal,decision_fct=NegoDecisionLogic,agent_decision_fct=BaseDecisionLogic,reward_fct=NegoRewardLogic,evaluation_fct=NegoEvaluationLogic,agent_type=NegoAgent,T=0,bidsplit=False,multibid=False):
        super().__init__(N=N,measurement_fct=measurement_fct,decision_fct=decision_fct,agent_decision_fct=agent_decision_fct,reward_fct=reward_fct,evaluation_fct=evaluation_fct,agent_type=agent_type,T=0)
        self.bidsplit=bidsplit
        self.multibid=multibid

    def __init_population(self,population=[]):
        super().__init_population(agent_type=NegoAgent)

    def decisions(self,perceptions=None):
        """
        Call the model's decision function, which would then query the individual agents if needed

        Kwargs:
        perceptions: the state of all agents in the population

        Returns:
        A list of actions of the same length as the population
        """
        partner = self.partner_set()
        return super().decisions(perceptions=perceptions)

    def partner_set(self):
        return self.decision_fct.get_partner(self.schedule.agents,bidsplit=self.bidsplit,multibid=self.multibid)
