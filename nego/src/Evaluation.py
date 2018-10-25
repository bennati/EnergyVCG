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
        # rewards=[i["reward"] for i in rewards]
        agents_low=[d for d in decisions if d['social_type']==1]
        agents_high=[d for d in decisions if d['social_type']==2]
        sellers_low=[d for d in agents_low if d['action']==1]
        sellers_high=[d for d in agents_high if d['action']==1]
        value_by_caste= lambda val,caste: sum([a.current_state['perception'][val] for a in self.model.schedule.agents if a.current_state['perception']['social_type']==caste])
        trade_by_caste =lambda caste_seller,caste_buyer: [
            sum([t for p,t in zip(a.current_state['partner'],a.current_state['transactions']) if p.current_state['perception']['social_type']==caste_buyer])
             for a in self.model.schedule.agents if (a.current_state['perception']['social_type']==caste_seller) & (a.current_state['action']==1)]
        tot_trade=sum(trade_by_caste(1,1)+trade_by_caste(1,2)+trade_by_caste(2,1)+trade_by_caste(2,2))
        assert(round(tot_trade,4)==round(sum([i for a in [x for x in self.model.schedule.agents if x.current_state['action']==1] for i in a.current_state['transactions']]),4)) # all transactions are captured
        assert(round(tot_trade,4)<=round(sum([x.current_state['perception']['initial_production'] for x in self.model.schedule.agents]),4))
        assert(round(tot_trade,4)<=round(sum([x.current_state['perception']['initial_consumption'] for x in self.model.schedule.agents]),4))
        div=lambda d,n: None if (n==0) | (n is None) | (d is None) else d/n
        def proportion(d,n):
            ret=div(d,n)
            if ret is not None:
                assert((round(ret,4)<=1) & (round(ret,4)>=0))
            return ret
        hd=proportion(sum(trade_by_caste(1,1)+trade_by_caste(2,1)),value_by_caste("initial_consumption",1))
        ho=proportion(sum(trade_by_caste(1,2)+trade_by_caste(2,2)),value_by_caste("initial_consumption",2))
        return [{
            "market_access":proportion(len(sellers_low)+len(sellers_high),len(decisions)),
            "market_access_low":proportion(len(sellers_low),len(agents_low)),
            "market_access_high":proportion(len(sellers_high),len(agents_high)),
            "trade_low_low":sum(trade_by_caste(1,1)),
            "trade_high_low":sum(trade_by_caste(2,1)),
            "trade_low_high":sum(trade_by_caste(1,2)),
            "trade_high_high":sum(trade_by_caste(2,2)),
            "sum_surplus_prod_low":value_by_caste("production",1),
            "sum_surplus_prod_high":value_by_caste("production",2),
            "sum_surplus_cons_low":value_by_caste("consumption",1),
            "sum_surplus_cons_high":value_by_caste("consumption",2),
            "sum_initial_prod_low":value_by_caste("initial_production",1),
            "sum_initial_prod_high":value_by_caste("initial_production",2),
            "sum_initial_cons_low":value_by_caste("initial_consumption",1),
            "sum_initial_cons_high":value_by_caste("initial_consumption",2),
            "satifaction_cons_low":hd,
            "satifaction_cons_high":ho,
            "satifaction_prod_low":proportion(sum(trade_by_caste(1,1)+trade_by_caste(1,2)),value_by_caste("initial_production",1)),
            "satifaction_prod_high":proportion(sum(trade_by_caste(2,1)+trade_by_caste(2,2)),value_by_caste("initial_production",2)),
            "efficiency":proportion(tot_trade,min(value_by_caste("initial_consumption",1)+value_by_caste("initial_consumption",2),value_by_caste("initial_production",1)+value_by_caste("initial_production",2))),
            "inequality":div(hd,ho)
        }]
