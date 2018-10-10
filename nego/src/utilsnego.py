import numpy as np
from scipy.stats import linregress
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

def efficiency_nego(all_efficiencies,tot_satisfied_agents):
    """
    calculates whether the demands are met exactly for the agents during transaction
    Args:
        all_efficiencies: efficiency of each agent's transaction
        tot_satisfied_agents: total agents who got their demand met

    Returns:
        either 1 if every agent gets their demand satisfied as needed or a fraction as per the demands satisfied
    """
    # print("total",tot_transactions)
    # print("traded",transaction_gap_ratio)
    # print("difference",tot_transactions-transaction_gap_ratio)
    return (sum(all_efficiencies)/tot_satisfied_agents if tot_satisfied_agents != 0 else 0)

def success_nego(tot_agents,tot_satisfied_agents):
    """
    calculates the ratio of number of agents who got allocated to partners
    Args:
        tot_agents: total agents who want to meet demands (are either seller or buyer)
        tot_transactions: total number of transactions in the round

    Returns:
        either 1 if all agents who need to buy/sell gets a partner or a fraction
        as per the number of agents getting partners
    """
    #print("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return (tot_satisfied_agents/tot_agents if tot_agents !=0 else 0)

def market_access(tot_agents,transactions):
    """
    calculates what fraction of agents traded
    Args:
        tot_agents: total number of agents who want to meet demands (are either seller or buyer)
        transactions: list of actions executed in the current round
    Returns:
        a fraction, related to the population size
    """
    #print("success=",(tot_transactions*2-tot_agents)/tot_agents)
    trades=[(t['partner'] is not None and t['action'] is not None) for t in transactions]
    return success_nego(tot_agents,sum(trades))

def fairness(measurements,decisions,N):
    """
    Args:
        decisions: list of decisions associated to every agent
        measurements: list of measurements associated to every agent
    Returns:
        relation coefficient based on the measurements of agents being related to the decisions they take
    """
    if len(measurements)>N:
        measurements = measurements[(len(measurements)-N):]
    if len(decisions)>N:
        decisions = decisions[(len(decisions)-N):]
    assert(len(measurements)==len(decisions))
    slope,intercept,rvalue,pvaue,stderr = linregress(measurements,decisions)
    print("fairness =", slope)
    return linregress(measurements,decisions)

def social_welfare(costs,rewards,N):
    """
    Computes the social welfare for the current round
    Args:
        costs: a list of costs, one for each agent
        rewards: a list of rewards, one for each agent
    Returns:
        the social welfare value
    """
    # s = np.mean(np.array(rewards))-np.mean(np.array(costs))
    s = np.mean(np.array(rewards))
    return s if s>0 else 0


def social_welfare_new(rewards):
    x = [i for i in rewards if i > 0]
    if np.count_nonzero(x)==1:
        s=0
    else:
        if x:
            s = min(i for i in rewards if i > 0)
        else:
            s = 0
    return s

def social_welfare_costs(costs,rewards,N):
    """
    Computes the social welfare for the current round
    Args:
        costs: a list of costs, one for each agent
        rewards: a list of rewards, one for each agent
    Returns:
        the social welfare value
    """
    s = np.mean(np.array(rewards))-np.mean(np.array(costs))
    # s = np.mean(np.array(rewards))
    return s if s>0 else 0

def is_mediator_biased(bias_mediator):
    '''
    Returns None if the mediator is not biased, or True with a probability equal to the parameter 'bias_mediator'
    '''
    return (np.random.uniform()<bias_mediator if bias_mediator!=0 else None) # the mediator is not biased if the bias is 0

def lo_hi(caste,lo,hi):
    return np.random.uniform()<(lo if caste else hi)
def is_productive(x,produce_low,produce_high):
    return lo_hi(x,produce_low,produce_high)
def is_biased(x,biased_low,biased_high):
    return lo_hi(x,biased_low,biased_high)

def individual_production(income,avg_production,caste,produce_low,produce_high):
    return (avg_production*income if is_productive(caste,produce_low,produce_high) else 0)

def split_bids(l,splitsize=1.0):
    """
    l: a list of bids, each bid is duplicates as many times as the bid is split, e.g. a bid of 3.1 is split in 4 (1,1,1,0.1).
    """
    return [{**s,'value':i} for s in l # create a new entry with the same records as the old dict, but with an updated value
                                for i in [1]*int(s['value']//splitsize)+[s['value']%splitsize]] # divide in bids of size splitsize

def reward_agent(decision):
    return (1 if decision["action"] is not None else 0)
