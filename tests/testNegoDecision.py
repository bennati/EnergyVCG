import unittest
from nego.src.DecisionLogic import *
from nego.src.Agent import NegoAgent

class TestNegoDecision(unittest.TestCase):
    """
    This class testes the measurement generation
    """

    def __init__(self, *args, **kwargs):
        super(TestNegoDecision, self).__init__(*args, **kwargs)
        self.dec=NegoDecisionLogic(None)
        self.getcons=lambda a: round(a.current_state['perception']['consumption'],3)
        self.getprod=lambda a: round(a.current_state['perception']['production'],3)

    def test_trade_biased_mediator(self):
        '''
        a biased mediator allows trading of individuals of the same caste and prevents trading of individuals of different castes
        '''
        buyer=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        buyer.current_state['perception'].update({'bias_mediator':True,'social_type':2})
        seller.current_state['perception'].update({'bias_mediator':True,'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))
        buyer.current_state['perception'].update({'bias_mediator':True,'social_type':1})
        assert(not self.dec.trade_allowed(seller,buyer))

    def test_trade_unbiased_mediator(self):
        """
        an unbiased mediator allows trading of individuals of different castes
        """
        buyer=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        seller.current_state['perception'].update({'bias_mediator':False})
        buyer.current_state['perception'].update({'bias_mediator':False})
        assert(self.dec.trade_allowed(seller,buyer))

    def test_trade_disable_mediator(self):
        """
        if the mediator is disabled, trade is conditioned by the agents' bias
        high-caste buyer bias prevents trading between castes, while low caste buyers are not subject to bias
        """
        buyer=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        buyer.current_state['perception'].update({'bias_mediator':None,'social_type':2,'biased':True})
        seller.current_state['perception'].update({'bias_mediator':None,'social_type':1})
        assert(not self.dec.trade_allowed(seller,buyer))
        buyer.current_state['perception'].update({'social_type':2,'biased':True})
        seller.current_state['perception'].update({'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))
        buyer.current_state['perception'].update({'social_type':1,'biased':True})
        seller.current_state['perception'].update({'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))

    def test_partner_selection(self):
        buyer=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        cons=np.random.uniform(10)
        prod=np.random.uniform(10)
        buyer.current_state['perception'].update({'consumption':cons,'bias_mediator':False}) # bias is disabled, trade is always possible
        buyer.current_state.update({'type':'buyer'})
        seller.current_state['perception'].update({'production':prod,'bias_mediator':False})
        seller.current_state.update({'type':'seller'})
        ans=self.dec.get_partner(population=[buyer,seller])
        assert([i['partner'] is not None for i in ans]) # all agents found a partner
        assert([i['partner']!=i['agent'] for i in ans]) # all agents found a valid partner
        ## test that values are updated correctly
        if cons>prod:
            assert(self.getcons(buyer)>=0) # rounding might cause it to be 0
            assert(self.getprod(seller)==0)
        elif prod>cons:
            assert(self.getcons(buyer)==0)
            assert(self.getprod(seller)>=0) # rounding might cause it to be 0
        else:
            assert(self.getcons(buyer)==0)
            assert(self.getprod(seller)==0)

    def test_partner_selection_multibid(self):
        """
        If multibid is enabled, a single individual can trade with multiple partners
        """
        buyer1=NegoAgent(0,None)
        buyer2=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        buyer1.current_state['perception'].update({'bias_mediator':False,"tariff":2.0}) # bias is disabled, trade is always possible
        buyer1.current_state.update({'type':'buyer'})
        buyer2.current_state['perception'].update({'bias_mediator':False,"tariff":1.0}) # bias is disabled, trade is always possible
        buyer2.current_state.update({'type':'buyer'})
        seller.current_state['perception'].update({'bias_mediator':False})
        seller.current_state.update({'type':'seller'})
        ## no multibid
        buyer1.current_state['perception'].update({'consumption':1.0})
        buyer2.current_state['perception'].update({'consumption':1.0})
        seller.current_state['perception'].update({'production':2.0})
        ans=self.dec.get_partner(population=[buyer1,buyer2,seller],multibid=False)
        assert(self.getcons(buyer1)==0)
        assert(self.getcons(buyer2)==1.0)
        assert(self.getprod(seller)==1.0)
        assert(buyer1.current_state['partner'][0]==seller)
        assert(buyer2.current_state['partner']==None)
        assert(seller.current_state['partner'][0]==buyer1)
        ## multibid
        buyer1.current_state['perception'].update({'consumption':1.0})
        buyer2.current_state['perception'].update({'consumption':1.0})
        seller.current_state['perception'].update({'production':2.0})
        ans=self.dec.get_partner(population=[buyer1,buyer2,seller],multibid=True)
        assert(self.getcons(buyer1)==0)
        assert(self.getcons(buyer2)==0.0)
        assert(self.getprod(seller)==0.0)
        assert(buyer1.current_state['partner'][0]==seller)
        assert(buyer2.current_state['partner'][0]==seller)
        assert(seller.current_state['partner'][0]==[buyer1,buyer2])

    def test_partner_selection_ordering(self):
        """
        agents are ordered based on their tariff
        """
        buyer1=NegoAgent(0,None)
        buyer2=NegoAgent(0,None)
        buyer3=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        buyer1.current_state['perception'].update({'bias_mediator':False}) # bias is disabled, trade is always possible
        buyer1.current_state.update({'type':'buyer'})
        buyer2.current_state['perception'].update({'bias_mediator':False}) # bias is disabled, trade is always possible
        buyer2.current_state.update({'type':'buyer'})
        buyer3.current_state['perception'].update({'bias_mediator':False}) # bias is disabled, trade is always possible
        buyer3.current_state.update({'type':'buyer'})
        seller.current_state['perception'].update({'bias_mediator':False})
        seller.current_state.update({'type':'seller'})
        ## buyers are ordered by descending tariff
        ## first ordering
        buyer1.current_state['perception'].update({'consumption':1.0,'tariff':3.0})
        buyer2.current_state['perception'].update({'consumption':1.2,'tariff':2.0})
        buyer3.current_state['perception'].update({'consumption':2.5,'tariff':1.0})
        seller.current_state['perception'].update({'production':2.0})
        ans=self.dec.get_partner(population=[buyer1,buyer2,buyer3,seller],multibid=True)
        assert(self.getcons(buyer1)==0)
        assert(self.getcons(buyer2)==0.2)
        assert(self.getcons(buyer3)==2.5)
        # second ordering
        buyer1.current_state['perception'].update({'consumption':1.0,'tariff':3.0})
        buyer2.current_state['perception'].update({'consumption':1.2,'tariff':1.0})
        buyer3.current_state['perception'].update({'consumption':2.5,'tariff':2.0})
        seller.current_state['perception'].update({'production':2.0})
        ans=self.dec.get_partner(population=[buyer1,buyer2,buyer3,seller],multibid=True)
        assert(self.getcons(buyer1)==0)
        assert(self.getcons(buyer2)==1.2)
        assert(self.getcons(buyer3)==1.5)
        ## reverse ordering
        buyer1.current_state['perception'].update({'consumption':1.0,'tariff':1.0})
        buyer2.current_state['perception'].update({'consumption':1.2,'tariff':2.0})
        buyer3.current_state['perception'].update({'consumption':2.5,'tariff':3.0})
        seller.current_state['perception'].update({'production':2.0})
        ans=self.dec.get_partner(population=[buyer1,buyer2,buyer3,seller],multibid=True)
        assert(self.getcons(buyer1)==1.0)
        assert(self.getcons(buyer2)==1.2)
        assert(self.getcons(buyer3)==0.5)

    def test_partner_selection_bidsplit(self):
        """
        Bids can be split into chunks
        """
        buyer1=NegoAgent(0,None)
        buyer2=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        buyer1.current_state['perception'].update({'bias_mediator':False,"tariff":3.0}) # bias is disabled, trade is always possible
        buyer1.current_state.update({'type':'buyer'})
        buyer2.current_state['perception'].update({'bias_mediator':False,"tariff":2.0}) # bias is disabled, trade is always possible
        buyer2.current_state.update({'type':'buyer'})
        seller.current_state['perception'].update({'bias_mediator':False})
        seller.current_state.update({'type':'seller'})
        # only multibid, the seller satisfies all demand
        buyer1.current_state['perception'].update({'consumption':1.1})
        buyer2.current_state['perception'].update({'consumption':1.1})
        seller.current_state['perception'].update({'production':3.01})
        ans=self.dec.get_partner(population=[buyer1,buyer2,seller],multibid=True,bidsplit=False)
        assert(self.getcons(buyer1)==0.0)
        assert(self.getcons(buyer2)==0.0)
        assert(self.getprod(seller)==0.81)
        # only bidsplit, bids of size <1 are paired, so there might be uncovered demand
        buyer1.current_state['perception'].update({'consumption':1.1})
        buyer2.current_state['perception'].update({'consumption':1.1})
        seller.current_state['perception'].update({'production':3.01})
        ans=self.dec.get_partner(population=[buyer1,buyer2,seller],multibid=False,bidsplit=True)
        assert(self.getcons(buyer1)==0.0)
        assert(self.getcons(buyer2)==0.09) # the corresponding production bid is worth only 0.01
        assert(self.getprod(seller)==0.9)
