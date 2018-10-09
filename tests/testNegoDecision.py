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

    def test_trade_biased_mediator(self):
        buyer=NegoAgent(0,None)
        seller=NegoAgent(0,None)
        # a biased mediator allows trading of individuals of the same caste
        buyer.current_state['perception'].update({'bias_mediator':True,'social_type':2})
        seller.current_state['perception'].update({'bias_mediator':True,'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))
        # a biased mediator prevents trading of individuals of different castes
        buyer.current_state['perception'].update({'bias_mediator':True,'social_type':1})
        assert(not self.dec.trade_allowed(seller,buyer))
        # an unbiased mediator allows trading of individuals of different castes
        seller.current_state['perception'].update({'bias_mediator':False})
        buyer.current_state['perception'].update({'bias_mediator':False})
        assert(self.dec.trade_allowed(seller,buyer))
        # if the mediator is disabled, trade is conditioned by the agents' bias
        buyer.current_state['perception'].update({'bias_mediator':None,'social_type':2,'biased':True})
        seller.current_state['perception'].update({'bias_mediator':None,'social_type':1})
        assert(not self.dec.trade_allowed(seller,buyer))
        # agents of the same caste can trade
        buyer.current_state['perception'].update({'social_type':2,'biased':True})
        seller.current_state['perception'].update({'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))
        # low caste buyers are not subject to bias
        buyer.current_state['perception'].update({'social_type':1,'biased':True})
        seller.current_state['perception'].update({'social_type':2})
        assert(self.dec.trade_allowed(seller,buyer))
