import numpy as np
from scipy.stats import linregress
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from src.utils import compute_stats,renormalize,positive_sampling
#rcParams.update({'figure.autolayout': True})

def efficiency_nego(population):
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
    # compute efficiency
    eff = []
    for a in population:
        if a.current_state["partner"] is not None:
            assert(a.get_decision() is not None)
            state=a.current_state["perception"]
            attr=("production" if a.current_state['type']=="seller"
                  else "consumption")              # buyer
            efficiency=1-(state[attr]/state["old_"+str(attr)]) # one if all needs are satisfied, a fraction otherwise
            eff.append(efficiency)
        else:
            eff.append(np.nan)
    tot_satisfied_agents=np.count_nonzero(~np.isnan(eff))
    return (np.nansum(eff)/tot_satisfied_agents if tot_satisfied_agents != 0 else np.nan)

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
    return (tot_satisfied_agents/tot_agents if tot_agents !=0 else np.nan)

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
    # TODO should we check market access based on the satisfaction of needs (production-old_production)?
    return success_nego(tot_agents,sum(trades))

# def fairness(measurements,decisions,N):
#     """
#     Args:
#         decisions: list of decisions associated to every agent
#         measurements: list of measurements associated to every agent
#     Returns:
#         relation coefficient based on the measurements of agents being related to the decisions they take
#     """
#     if len(measurements)>N:
#         measurements = measurements[(len(measurements)-N):]
#     if len(decisions)>N:
#         decisions = decisions[(len(decisions)-N):]
#     assert(len(measurements)==len(decisions))
#     slope,intercept,rvalue,pvaue,stderr = linregress(measurements,decisions)
#     print("fairness =", slope)
#     return linregress(measurements,decisions)

# def social_welfare(costs,rewards,N):
#     """
#     Computes the social welfare for the current round
#     Args:
#         costs: a list of costs, one for each agent
#         rewards: a list of rewards, one for each agent
#     Returns:
#         the social welfare value
#     """
#     # s = np.mean(np.array(rewards))-np.mean(np.array(costs))
#     s = np.mean(np.array(rewards))
#     return s if s>0 else 0


# def social_welfare_new(rewards):
#     x = [i for i in rewards if i > 0]
#     if np.count_nonzero(x)==1:
#         s=0
#     else:
#         if x:
#             s = min(i for i in rewards if i > 0)
#         else:
#             s = 0
#     return s

# def social_welfare_costs(costs,rewards,N):
#     """
#     Computes the social welfare for the current round
#     Args:
#         costs: a list of costs, one for each agent
#         rewards: a list of rewards, one for each agent
#     Returns:
#         the social welfare value
#     """
#     s = np.mean(np.array(rewards))-np.mean(np.array(costs))
#     # s = np.mean(np.array(rewards))
#     return s if s>0 else 0

def social_welfare_rawls(rewards):
    return (np.nan if all(np.isnan(rewards)) else min(np.array(rewards)[~np.isnan(rewards)]))

def is_mediator_biased(bias_mediator):
    '''
    Returns None if the mediator is not biased, or True with a probability equal to the parameter 'bias_mediator'
    '''
    return (None if bias_mediator is None else np.random.uniform()<bias_mediator) # the mediator is not biased if the bias is 0

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
                                for i in [1]*int(s['value']//splitsize)+([s['value']%splitsize] if s['value']%splitsize!=0 else [])] # divide in bids of size splitsize

def reward_agent(decision):
    return (1 if decision["action"] is not None else np.nan) # TODO, consider lack of rewards? set 0 instead of np.nan

def compute_incomes(df,castes):
    f=lambda caste,value,binno=None: df[(df['caste']==("Dalit" if caste else "Other")) & (True if binno is None else df['income_min']==binno)][value]
    bins=[np.random.choice(f(c,'income_min'), # determine income level randomly
                           p=f(c,'value_mean')/f(c,'value_mean').sum() # from the probability that a given caste has a given income level
    ) for c in castes]
    return [int(np.random.uniform(b,float(f(c,'income_max',binno=b)))) for b,c in zip(bins,castes)]

def read_income_data(datadir):
    byincome=pd.read_csv(os.path.join(datadir,"income_byincome.csv")).drop("Per capita income category (Rs per annum)",axis=1)
    byvillage=pd.read_csv(os.path.join(datadir,"income_byvillage.csv")) # contains the proportion of dalits in village
    ## aggregate villages
    index=["income_min","income_max"]
    byincome=pd.melt(byincome,id_vars=index) # make the table long
    byincome["village"],byincome["caste"]=byincome["variable"].str.split(' ',1).str # split village and caste
    ## multiply proportions of dalits in each income level by their proportion in the village
    byincome['value_norm']=byincome[['value','village','caste']].apply(lambda x: x.value/100.0*float(byvillage.loc[byvillage["Village"]==x.village,x.caste+"_prop"]),axis=1)
    return compute_stats(byincome,idx=index+["caste"],columns=["value","value_norm"]),byvillage

def compute_consumptions(consumption_data,deviation,incomes,max_income,avg_cons_india=805.599,avg_cons_usa=12984.333,btu_2_kwh=0.000293071): #2005 data: avg_cons_india=469.454,avg_cons_usa=13704.577
    incomes_usd=[renormalize(i,[0,max_income],[0,consumption_data.income_max.max()])[0] for i in incomes]
    consumptions=[float(consumption_data.loc[(consumption_data.income_min<=i) & (consumption_data.income_max>=i),'Per household (million Btu)']) for i in incomes_usd]
    consumptions=[c/365                  # get daily value
                  *10e6*btu_2_kwh        # convert to kWh
                  *avg_cons_india/avg_cons_usa # rescale for indian market
                  for c in consumptions]
    return [positive_sampling(c,deviation) for c in consumptions]

def compute_productions(incomes,yearly_disposable_income=0.2,installment_cost=1600,device_production=0.1):
    """
    Individuals must be able to afford the cost of the equipment to produce energy.
    Survey data shows that the households belonging to the bottom quintiles spends around 20\% on other expenses, which we consider disposable income to pay for electricity production.
    Given that a device with a production of 0.1kWh costs around 1600 Rupees and the lifespan of a solar panel is around 20 years, we assume households invest all of their disposable income for the following 20 years to buy as many devices as they can afford.
    """
    return [i*yearly_disposable_income*20 # total disposable income over lifespan of panel
            //installment_cost               # how many panels one can afford
            *device_production              # convert to kWh
            for i in incomes] # the affordable device produces this many kWh

def bias_fct_divide_castes(seller, buyer):
    s=seller.current_state["perception"]
    b=buyer.current_state["perception"]
    assert(s["bias_mediator"]==b["bias_mediator"]) # there is only one mediator
    mediator_biased=s["bias_mediator"] # boolean or None if the mediator is not biased
    if mediator_biased is None: # the mediator does not influence trading, the agents determine the outcome of the transaction
        cantrade=not (b["biased"] and b["social_type"]==2 and s["social_type"]==1) # buyers of high caste don't want to trade with low caste, sellers can trade with anyone
    elif mediator_biased: # the mediator is biased, it determines the outcome of the transaction
        cantrade=b["social_type"]==s["social_type"]
    else:                   # the mediator is not biased, the trade can take place
        cantrade=True
    return cantrade

def bias_fct_mediator_equals_agents(seller, buyer):
    s=seller.current_state["perception"]
    b=buyer.current_state["perception"]
    assert(s["bias_mediator"]==b["bias_mediator"]) # there is only one mediator
    mediator_biased=s["bias_mediator"] # boolean or None if the mediator is not biased
    if mediator_biased is None: # the mediator does not influence trading, the agents determine the outcome of the transaction
        cantrade=not (b["biased"] and b["social_type"]==2 and s["social_type"]==1) # buyers of high caste don't want to trade with low caste, sellers can trade with anyone
    elif mediator_biased: # the mediator is biased, it determines the outcome of the transaction
        cantrade=not (b["social_type"]==2 and s["social_type"]==1) # buyers of high caste don't want to trade with low caste, sellers can trade with anyone
    else:                   # the mediator is not biased, the trade can take place
        cantrade=True
    return cantrade
