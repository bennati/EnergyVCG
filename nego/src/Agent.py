from src.Agent import BaseAgent
from nego.src.DecisionLogic import NegoDecisionLogicAgent
import operator
import numpy as np

class NegoAgent(BaseAgent):
    def __init__(self,unique_id,model,decision_fct=NegoDecisionLogicAgent):
        """
        The state contains:
        action: last action, 0 for buy 1 for sell
        reward: last reward obtained by the supervisor
        tariff: the preferred prico of sale
        type: whether the agent is a seller or a buyer in this turn
        partner: the selected partner
        perception: last measurement, containing:
            production: how much energy is produced
            consumption: how much energy is required
            cost: the cost of contribution
        """
        super().__init__(unique_id,model,decision_fct=decision_fct)
        self.current_state={"perception":{"production":0,"consumption":0,"tariff":0,"social_type":0},"type":None,
                            "partner":None,"transactions":None,"action":0,"cost":0,"reward":{"agentID":self.unique_id,"reward":0},"agentID":self.unique_id}

    def seller_buyer(self):
        """

        Returns: whether an agent is a seller or a buyer, or None if production and consumption are 0

        """
        self.current_state.update({"type":None})
        state=self.current_state["perception"]
        if (state["production"] > state["consumption"]) and state["production"]!=0:
            self.current_state['perception'].update({"consumption":0}) # either seller or buyer
            self.current_state.update({"type":"seller"})
            # self.current_state.update({"cost":self.current_state["perception"]["production"]*
            #                            self.current_state["perception"]["main_cost"]}) # update cost for sellers
        if (state["production"] < state["consumption"]) and state["consumption"]!=0:
            self.current_state['perception'].update({"production":0}) # either seller or buyer
            self.current_state.update({"type":"buyer"})
        return self.current_state["type"]

    def step(self):
        self.current_state["action"]=None
        self.current_state["partner"]=None
        self.seller_buyer()
        self.current_state["perception"]["initial_production"]=self.current_state["perception"]["production"]
        self.current_state["perception"]["initial_consumption"]=self.current_state["perception"]["consumption"]
        super().step()
