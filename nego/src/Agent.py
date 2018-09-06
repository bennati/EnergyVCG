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
                            "partner":None,"action":0,"cost":0,"reward":0,"agentID":self.unique_id}

    def seller_buyer(self):
        """

        Returns: whether an agent is a seller or a buyer, or None if production and consumption are 0

        """
        state=self.current_state["perception"]
        if state["production"] > state["consumption"] and state["production"]!=0:
            self.current_state['perception'].update({"consumption":0}) # either seller or buyer
            self.current_state.update({"type":"seller"})
            self.current_state.update({"cost":self.current_state["perception"]["production"]*
                                       self.current_state["perception"]["main_cost"]}) # update cost for sellers
        if state["production"] < state["consumption"] and state["consumption"]!=0:
            self.current_state['perception'].update({"production":0}) # either seller or buyer
            self.current_state.update({"type":"buyer"})
        return self.current_state["type"]

    # def partner_selection(self):
    #     """

    #     Returns: the partner for bilateral based on simply matching any seller with any buyer

    #     """
    #     other = self.model.schedule.agents
    #     perc = self.current_state["perception"]
    #     self.seller_buyer()
    #     for a in other:
    #         if a != self:
    #             perc_other = a.current_state["perception"]
    #             if self.current_state["type"] != a.current_state["type"]:
    #                 self.current_state["partner"] = a
    #     return self.current_state["partner"]

    # def partner_selection_orderbid(self):
    #     """

    #     Returns: partner for bilateral matching using ordering bids logic

    #     """
    #     other = self.model.schedule.agents
    #     perc=self.current_state["perception"]
    #     sellers = []
    #     buyers = []
    #     for a in other:
    #         a.seller_buyer()
    #         perc_other = a.current_state["perception"]
    #         if a.current_state["type"] == "seller":
    #             sellers.append({"agent":a,"agent_bid":perc_other["tariff"],
    #                             "value":a.current_state["perception"]["production"]})
    #         elif a.current_state["type"] == "buyer":
    #             buyers.append({"agent":a,"agent_bid":perc_other["tariff"],
    #                            "value":a.current_state["perception"]["consumption"]})
    #     sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))  # ascending sorted sellers as bids
    #     buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)  # descending sorted buyers as bids
    #     i=0
    #     j=0
    #     len_s=len(sellers_sorted)
    #     len_b=len(buyers_sorted)
    #     r = min(len_b,len_s)
    #     while True:
    #         if i<r and j<r and r>=1:
    #             if sellers_sorted[i]["value"] !=0 and buyers_sorted[j]["value"]!=0:
    #                 k = (sellers_sorted[i]["value"])-(buyers_sorted[j]["value"])
    #                 if k==0:
    #                     x = sellers_sorted[i]["agent"]
    #                     y = buyers_sorted[i]["agent"]
    #                     x.current_state.update({"partner":y})
    #                     y.current_state.update({"partner":x})
    #                     x.current_state["perception"].update({"production":0})
    #                     y.current_state["perception"].update({"consumption":0})
    #                     sellers_sorted[i]["value"] = 0
    #                     buyers_sorted[j]["value"] = 0
    #                     j+=1
    #                     i+=1
    #                 if k>0:
    #                     x = sellers_sorted[i]["agent"]
    #                     y = buyers_sorted[i]["agent"]
    #                     x.current_state.update({"partner":y})
    #                     y.current_state.update({"partner":x})
    #                     y.current_state["perception"].update({"old_consumption":
    #                                                               y.current_state["perception"]["consumption"]})
    #                     x.current_state["perception"].update({"production":
    #                                                               x.current_state["perception"]["production"]-
    #                                                               y.current_state["perception"]["consumption"]})
    #                     sellers_sorted[i]["value"] = sellers_sorted[i]["value"]-buyers_sorted[j]["value"]
    #                     buyers_sorted[j]["value"] = 0
    #                     j+=1
    #                     if sellers_sorted[i]["value"]<=0:
    #                         i+=1
    #                 if k<0:
    #                     x = sellers_sorted[i]["agent"]
    #                     y = buyers_sorted[j]["agent"]
    #                     x.current_state.update({"partner":y})
    #                     y.current_state.update({"partner":x})
    #                     x.current_state["perception"].update({"old_production":
    #                                                               x.current_state["perception"]["production"]})
    #                     y.current_state["perception"].update({"consumption":
    #                                                               y.current_state["perception"]["consumption"]-
    #                                                               x.current_state["perception"]["production"]})
    #                     buyers_sorted[j]["value"] = buyers_sorted[j]["value"]-sellers_sorted[i]["value"]
    #                     sellers_sorted[i]["value"] = 0
    #                     i+=1
    #                     if buyers_sorted[j]["value"]<=0:
    #                         j+=1
    #             else:
    #                 break
    #         else:
    #             break
    #     # if len(sellers_sorted)<=len(buyers_sorted): # the remaining energy is wasted
    #     #     sorted_list = sellers_sorted
    #     #     other_list = buyers_sorted
    #     # else:
    #     #     sorted_list = buyers_sorted
    #     #     other_list = sellers_sorted
    #     # for i in range(len(sorted_list)):
    #     #     x = sorted_list[i]["agent"]
    #     #     y = other_list[i]["agent"]
    #     #     if (x.current_state["type"]=="buyer" and x.current_state["perception"]["consumption"]<
    #     #             y.current_state["perception"]["production"]-y.current_state["perception"]["consumption"]) or \
    #     #             (x.current_state["type"]=="seller" and y.current_state["perception"]["consumption"]<
    #     #                     x.current_state["perception"]["production"]-x.current_state["perception"]["consumption"]):
    #     #         x.current_state.update({"partner":y})
    #     #         y.current_state.update({"partner":x})
    #     return self.current_state["partner"]

    # def transactions(self):
    #     """

    #     Returns: cost for the sellers for their transactions/ energy trade

    #     """
    #     if self.current_state["type"] == "seller":
    #         if self.current_state["perception"]["production"] != 0:
    #             self.current_state.update({"cost":self.current_state["perception"]["production"]*
    #                                               self.current_state["perception"]["main_cost"]})
    #     return self.current_state["cost"]

    def step(self):
        self.current_state["action"]=np.nan
        self.current_state["partner"]=None
        self.seller_buyer()
        self.current_state["perception"]["old_production"]=self.current_state["perception"]["production"]
        self.current_state["perception"]["old_consumption"]=self.current_state["perception"]["consumption"]
        super().step()
