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
        partners = self.partner_set()
        # debug
        cons=[a.current_state['perception']['consumption'] for a in self.schedule.agents]
        prods=[a.current_state['perception']['production'] for a in self.schedule.agents]
        ## update agents states
        for a in self.schedule.agents:
            l=[x for x in partners if x['agent']==a] # isolate the entries that correspond to the agent
            assert(len(l)>0)
            act=l[0]['action']
            assert([x['action']==act for x in l]) # all actions are the same
            partner=l[0]['partner']
            assert([x['partner']==partner for x in l]) # all partners are the same
            a.current_state.update({"action":act,"partner":partner})
            if act is not None:
                a.current_state['perception'].update({("production" if act==1 else "consumption"):sum([x['value'] for x in l])}) # sum all contributions in dict
        # debug
        cons_new=[a.current_state['perception']['consumption'] for a in self.schedule.agents]
        prods_new=[a.current_state['perception']['production'] for a in self.schedule.agents]
        assert(round(sum(cons)-sum(prods),4)==round(sum(cons_new)-sum(prods_new),4))
        return super().decisions(perceptions=perceptions)

    def partner_set(self):
        return self.decision_fct.get_partner(self.schedule.agents,bidsplit=self.bidsplit,multibid=self.multibid)
