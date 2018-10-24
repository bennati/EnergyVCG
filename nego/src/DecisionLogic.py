from src.DecisionLogic import BaseDecisionLogic
from nego.src.utilsnego import *
import operator
import math
import numpy as np

class NegoDecisionLogic(BaseDecisionLogic):
    def __init__(self, model,bias_fct=bias_fct_divide_castes):
        super().__init__(model)
        self.bias_fct=bias_fct
    def get_decision(self,perceptions):
        """
        Args:
        perceptions: a list of dictionaries, each containing the perception vector associated with one agent
        Returns: a list of dictionaries representing the contributions of the agents.
        Must contain keys ["agentID", "timestep"]
        """
        self.act=[{"production":p["production"],"consumption":p["consumption"],
                   "tariff":p["tariff"],"agentID":a.unique_id,
                   # "cost":a.current_state["cost"],
                   "reward":a.current_state["reward"]['reward'],"action":a.current_state["action"],
                   "partner":(None if a.current_state["partner"] is None else [x.unique_id for x in a.current_state["partner"]]),
                   "social_type_partner":(None if a.current_state["partner"] is None else [x.current_state['perception']['social_type'] for x in a.current_state["partner"]]),
                   "social_type":p["social_type"],
                   "biased":p["biased"],"bias_mediator":p["bias_mediator"]}
                  for a,p in zip(self.model.schedule.agents,perceptions)]
        return self.act

    def update_partner(self,action,agent,partner):
        agent.update({"action":action,"partner":([partner["agent"]
                                                  if agent["partner"] is None # first transactions
                                                  else agent["partner"]+[partner["agent"]]])})
        return agent

    def get_partner(self,population,bidsplit=False,multibid=False):
        sellers = []
        buyers = []
        ## divide buyers and sellers
        for a in population:
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"action":None,"partner":None,"value":a.current_state["perception"]["production"],"agent_bid":a.current_state["perception"]["tariff"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"action":None,"partner":None,"value":a.current_state["perception"]["consumption"],"agent_bid":a.current_state["perception"]["tariff"]})
            elif a.current_state["type"] is None:
                pass
            else:
                raise ValueError("Wrong type: "+str(a.current_state["type"]))
        if bidsplit: ## split in smaller bids, duplicate dict entries with different bids
            sellers=split_bids(sellers); buyers=split_bids(buyers)
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid')) # ascending
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True) # descending
        # print(str(len(sellers))+" sell and "+str(len(buyers))+" buy")
        i=0
        j=0
        while i<len(sellers_sorted) and j<len(buyers_sorted):
            seller=sellers_sorted[i]
            buyer=buyers_sorted[j]
            i+=1; j+=1 # move to next buyer and seller. If the trade is biased, skip current buyer and seller
            sv=seller["value"]
            bv=buyer["value"]
            if self.bias_fct(seller["agent"],buyer["agent"]): # trade is not prevented
                # print("-------------------- SELLERS "+str([p["value"] for p in sellers_sorted]))
                # print("-------------------- BUYERS "+str([p["value"] for p in buyers_sorted]))
                if sv !=0 and bv !=0:  # can trade
                    ## update dictionary with new actions and partners
                    seller=self.update_partner(1,seller,buyer) # sell
                    buyer=self.update_partner(2,buyer,seller) # buy
                    k = sv-bv                                 # mismatch between demand and offer
                    ## update values in dictionary
                    if k==0:    # perfect match
                        seller["value"]=0; buyer["value"]=0
                    if k>0:     # seller offers more
                        seller["value"]=k; buyer["value"]=0
                        if multibid:
                            i-=1 # keep current seller, which has more to trade
                        # print("NEW PROD "+str(seller["value"]))
                    if k<0:     # buyer wants more
                        seller["value"]=0; buyer["value"]=-k
                        if multibid:
                            j-=1 # keep current buyer, which has more to trade
                        # print("NEW CONS "+str(buyer["value"]))
                else:
                    print("?????????????????????????????? BREAK")
                    print("seller "+str(sv)+" "+str(sv!=0))
                    print("buyer "+str(bv)+" "+str(bv!=0))
                    break
            # else:
            #     print("BIASSSSS")
        # print("-------------------- LAST SELLERS "+str([p["value"] for p in sellers_sorted]))
        # print("-------------------- LAST BUYERS "+str([p["value"] for p in buyers_sorted]))
        # return [({"agent":a,"partner":a.current_state["partner"]}) for a in population]
        return sellers_sorted+buyers_sorted

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """

    def get_decision(self,perceptions):
        return self.model.current_state["action"]

    def feedback(self,perceptions,reward):
        rew = self.model.current_state["reward"]
        rew1 = self.model.current_state["perception"]
        partner = self.model.current_state["partner"]
        if self.model.current_state["type"]=="seller":
            if partner!=None:
                assert(rew1["initial_production"]-rew1["production"]>=0) # production cannot increase
                rew.update({"reward":(rew1["initial_production"]-rew1["production"])*2})
        return self.model.current_state["reward"]
