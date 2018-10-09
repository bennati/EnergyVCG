from src.DecisionLogic import BaseDecisionLogic
import operator
import math
import numpy as np

class NegoDecisionLogic(BaseDecisionLogic):
    def get_decision(self,perceptions):
        """
        Args:
        perceptions: a list of dictionaries, each containing the perception vector associated with one agent
        Returns: a list of dictionaries representing the contributions of the agents.
        Must contain keys ["agentID", "timestep"]
        """
        # decs=[a.decisions(p) for a,p in zip(self.model.schedule.agents,perceptions)]
        # call each agent's decision fct with the appropriate perception
        # partner = self.model.partner_selection_orderbid()
        self.act=[{"production":p["production"],"consumption":p["consumption"],
                   "tariff":p["tariff"],"agentID":a.unique_id,
                   "contribution":0,"contributed":False,"cost":a.current_state["cost"], # TODO why is contribution always 0?
                   "reward":a.current_state["reward"],"action":a.current_state["action"],
                   "partner":a.current_state["partner"],"social_type":p["social_type"],
                   "biased":p["biased"],"bias_mediator":p["bias_mediator"]}
                  for a,p in zip(self.model.schedule.agents,perceptions)]
        return self.act

    def trade_allowed(self,seller, buyer):
        s=seller.current_state["perception"]
        b=buyer.current_state["perception"]
        assert(s["bias_mediator"]==b["bias_mediator"]) # there is only one mediator
        # TODO should the mediator bias influence also the partner selection?
        mediator_biased=s["bias_mediator"] # boolean or None if the mediator is not biased
        if mediator_biased is None: # the mediator does not influence trading, the agents determine the outcome of the transaction
            cantrade=not (b["biased"] and b["social_type"]==2 and s["social_type"]==1) # buyers of high caste don't want to trade with low caste, sellers can trade with anyone
        elif mediator_biased: # the mediator is biased, it determines the outcome of the transaction
            cantrade=b["social_type"]==s["social_type"]
        else:                   # the mediator is not biased, the trade can take place
            cantrade=True
        return cantrade

    def update_partner(self,action,agent,partner):
        agent.update({"action":action,"partner":([partner["agent"]
                                                  if agent["partner"] is None # first transactions
                                                  else agent["partner"]+[partner["agent"]]])})
        return agent

    def get_partner(self,bidsplit=False,multibid=False):
        sellers = []
        buyers = []
        ## divide buyers and sellers
        for a in self.model.schedule.agents:
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"action":np.nan,"partner":None,"value":a.current_state["perception"]["production"],"agent_bid":a.current_state["perception"]["tariff"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"action":np.nan,"partner":None,"value":a.current_state["perception"]["consumption"],"agent_bid":a.current_state["perception"]["tariff"]})
            else:
                print(a.current_state["type"])
        if bidsplit:
            ## split in smaller bids, duplicate dict entries with different bids
            splitsize=1.0
            split_entries=lambda l: [{**s,'value':i} for s in l # create a new entry with the same records as the old dict, but with an updated value
                                for i in [1]*int(s['value']//splitsize)+[s['value']%splitsize]] # divide in bids of size splitsize
            sellers=split_entries(sellers); buyers=split_entries(buyers)
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
            if self.trade_allowed(seller["agent"],buyer["agent"]): # trade is not prevented
                # print("-------------------- SELLERS "+str([p["value"] for p in sellers_sorted]))
                # print("-------------------- BUYERS "+str([p["value"] for p in buyers_sorted]))
                if sv !=0 and bv!=0:  # can trade
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
                            print(j)
                        # print("NEW CONS "+str(buyer["value"]))
                else:
                    print("?????????????????????????????? BREAK")
                    print(sv)
                    print(bv)
                    break
            # else:
            #     print("BIASSSSS")
        # print("-------------------- LAST SELLERS "+str([p["value"] for p in sellers_sorted]))
        # print("-------------------- LAST BUYERS "+str([p["value"] for p in buyers_sorted]))
        ## update agents states
        for a in self.model.schedule.agents:
            ## aggregate values belonging to the same user, used in case of bidsplit
            def update_agent_state(agent,dct,varname):
                l=[x for x in dct if x['agent']==agent] # isolate the entries that correspond to the agent
                if l!=[]:
                    act=l[0]['action']
                    assert([x['action']==act for x in l]) # all actions are the same
                    partner=l[0]['partner']
                    assert([x['partner']==partner for x in l]) # all partners are the same
                    agent.current_state.update({"action":act,"partner":partner})
                    # if agent.current_state['perception'][varname]!=sum([x['value'] for x in l]):
                    #     print("Updating "+varname+" from "+str(agent.current_state['perception'][varname])+" to "+str([x['value'] for x in l]))
                    agent.current_state['perception'].update({varname:sum([x['value'] for x in l])}) # sum all contributions in dict
            update_agent_state(a,sellers_sorted,'production')
            update_agent_state(a,buyers_sorted,'consumption')
        # print("-------------------- PARTNERS "+str([a.current_state["partner"] for a in self.model.schedule.agents]))
        return [({"agent":a,"partner":a.current_state["partner"]}) for a in self.model.schedule.agents]

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """

    def get_decision(self,perceptions):
        print("aodhpoba;sjb;jb;absd;jd;vododododo")
        return self.model.current_state["action"]

    def feedback(self,perceptions,reward):
        rew = self.model.current_state["reward"]
        rew1 = self.model.current_state["perception"]
        partner = self.model.current_state["partner"]
        if self.model.current_state["type"]=="seller":
            if partner!=None:
                assert(rew1["old_production"]-rew1["production"]>=0) # production cannot increase
                rew.update({"reward":(rew1["old_production"]-rew1["production"])*2})
        return self.model.current_state["reward"]
