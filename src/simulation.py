import functools
import itertools
import pandas as pd
import numpy as np
from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from MeasurementGen import BaseMeasurementGen
from utils import *
from Agent import *
from Supervisor import *

def run_experiment(test,conf):
    log_tot=[]
    for r in range(conf["reps"]):
        for idx,p in expandgrid(conf["params"]).iterrows():
            params=p.to_dict()
            params.update({"repetition":r})
        f=functools.partial(conf["meas_fct"],**params)
            model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=RewardLogicUniform,agent_type=BaseAgent)
            model.run(conf["T"],params=params)
        log_tot=log_tot+model.log # concatenate lists
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        stats_rew=get_stats(log_tot,"reward",idx=[varname],cols=["reward"])
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["value","cost"])
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["contribution","cost","contributed"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","cost","efficiency","social_welfare","success","tot_contrib"])
        plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+".pdf")
        plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+".pdf")


class RewardLogicFull(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        Almost full contribution is required
        """
        percs=np.sum([p["value"] for p in self.model.current_state["perception"]])
        thresh=np.random.uniform(percs*0.8,percs) # almost full contrib
        contribs=np.sum([d["contribution"] for d in decisions])
        outcome=success(thresh,np.sum(contribs))
        if outcome==1:
            costs=np.array([d["cost"] for d in decisions])
            ret=-costs+self.benefit
            ret=[{"reward":r} for r in ret]
        else:
            ret=[{"reward":self.damage}]*self.model.N
        return ret

class RewardLogicUniform(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=5
        self.damage=-10

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in self.model.current_state["perception"]])
        contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
        # if thresh<=contribs:
        #     print("success "+str(thresh)+" "+str(contribs))
        # else:
        #     print("insuccess "+str(thresh)+" "+str(contribs))
        outcome=success(thresh,contribs)
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
        if outcome==1:
            ret=-costs+self.benefit
        else:
            print("unsuccessful")
            ret=-costs+self.damage
        ret=[{"reward":r} for r in ret]
        return ret

class DecisionLogicEmpty(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        pass

    def feedback(self,perceptions,reward):
        pass

class DecisionLogicSupervisorMandatory(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],"contributed":True,"timestep":a["timestep"]} for a in perceptions]
        return self.last_actions

def knapsack(items, maxweight):
    """
    https://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` that weighs no more than `maxweight`.

    `items` is a sequence of pairs `(value, weight)`, where `value` is
    a number and `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])
    """
    N = len(items)
    W = maxweight
    # Create an (N+1) by (W+1) 2-d list to contain the running values
    # which are to be filled by the dynamic programming routine.
    # bestvalues[i][j] is the best sum of values for any
    # subsequence of the first i items, whose weights sum
    # to no more than j.
    bestvalues = [[0] * (maxweight + 1)
                  for _ in range(len(items) + 1)]

    # Enumerate through the items and fill in the best-value table
    for i, (agentID,value, weight) in enumerate(items):
        for capacity in range(maxweight + 1):
            # Handle the case where the weight of the current item is greater
            # than the "running capacity" - we can't add it to the knapsack
            if weight > capacity:
                bestvalues[i+1][capacity] = bestvalues[i][capacity]
            else:
                # Otherwise, we must choose between two possible candidate values:
                # 1) the value of "running capacity" as it stands with the last item
                #    that was computed; if this is larger, then we skip the current item
                # 2) the value of the current item plus the value of a previously computed
                #    set of items, constrained by the amount of capacity that would be left
                #    in the knapsack (running capacity - item's weight)
                candidate1 = bestvalues[i][capacity]
                candidate2 = bestvalues[i][capacity - weight] + value

                # Just take the maximum of the two candidates; by doing this, we are
                # in effect "setting in stone" the best value so far for a particular
                # prefix of the items, and for a particular "prefix" of knapsack capacities
                bestvalues[i+1][capacity] = max(candidate1, candidate2)

    # Reconstruction
    # Iterate through the values table, and check
    # to see which of the two candidates were chosen. We can do this by simply
    # checking if the value is the same as the value of the previous row. If so, then
    # we say that the item was not included in the knapsack (this is how we arbitrarily
    # break ties) and simply move the pointer to the previous row. Otherwise, we add
    # the item to the reconstruction list and subtract the item's weight from the
    # remaining capacity of the knapsack. Once we reach row 0, we're done
    reconstruction = []
    j = maxweight
    for i in range(N, 0, -1):
        if bestvalues[i][j] != bestvalues[i - 1][j]:
            reconstruction.append(items[i - 1])
            j -= items[i - 1][2] # subtract capacity

    # Reverse the reconstruction list, so that it is presented
    # in the order that it was given
    reconstruction.reverse()

    # Return the best value, and the reconstruction list
    return bestvalues[len(items)][maxweight], reconstruction

class DecisionLogicSupervisorKnapsack(BaseDecisionLogic):
    """
    Optimize the knapsack problem
    """
    def get_decision(self,perceptions):
        W=max([a["threshold"] for a in perceptions]) # public good threshold
        items=[(a["agentID"],a["cost"],a["value"]) for a in perceptions] # values to maximize and contributions to achieve
        maxcost=max([c for i,c,v in items])
        items=[(i,maxcost-c+1,v) for i,c,v in items] # invert the costs because the knapsack is a maximization problem
        _,vals=knapsack(items,W)
        s=sum([v for i,c,v in vals])
        assert(s<=W)
        excluded=[]
        tmp=vals.copy()
        for i in items: # find users that do not contribute
            if i not in tmp:
                excluded.append(i)
            else:
                tmp.remove(i) # remove the value
        del tmp
        # add an additional contributor in case the threshold is not met
        if(excluded != []       # if not all users contributed
           and s<W):              # and if another contributor is needed
            vals.append(max(excluded, key=lambda x: x[1])) # add the contributor with the lowest cost
        assert(sum([v for i,c,v in vals])>=W)           # successful
        # find what agents contribute
        idx=[i for i,c,v in vals] # take first occurrence, it works even if two agents have the same cost and value
        decisions=[True if i in idx else False for i in range(len(perceptions))]
        assert(sum(decisions)==len(idx))
        assert(sum([perceptions[i]["value"] for i in idx])>=W)
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],"contributed":d,"timestep":a["timestep"]} for a,d in zip(perceptions,decisions)]
        return self.last_actions

class DecisionLogicSupervisorProbabilistic(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=[{"contribution":a["value"],"cost":a["cost"],"agentID":a["agentID"],"contributed":(True if np.random.uniform()<=0.5 else False),"timestep":a["timestep"]} for a in perceptions]
        return self.last_actions

class MeasurementGenUniform(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=kwargs["n1"]
        self.n2=kwargs["n2"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[max(1,np.random.randint(self.n1,self.n2)) for _ in population]
        # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
        thresh=np.random.randint(1,10)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":max(1,np.random.randint(1,4)),"timestep":timestep,"agentID":i,"threshold":thresh} for i,v in enumerate(vals)]
        return ret

class MeasurementGenNormal(BaseMeasurementGen):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.mu=kwargs["mu"]
        self.s=3

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"value":max(0.01,np.random.normal(loc=self.mu,scale=self.s)),"cost":0,"timestep":timestep,"agentID":i} for i in range(len(population))]
        return ret

class MeasurementGenBinomial(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=1
        self.sep=kwargs["rich"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"value":(max(0.01,np.random.normal(loc=self.mu1,scale=self.s1))
                       if i>len(population)*self.sep else
                       max(0.01,np.random.normal(loc=self.mu2,scale=self.s2))),"cost":0,"timestep":timestep,"agentID":i} for i in range(len(population))]
        return ret

if __name__ == '__main__':
    # tests={"uniform":{"T":10,"reps":2,"params":{"N":[10],"mu":[5,20,50]},"meas_fct":MeasurementGenNormal,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty},"binomial":{"T":10,"reps":2,"params":{"N":[10],"mu1":[1],"mu2":[5,20,50],"rich":[0.2,0.5,0.8]},"meas_fct":MeasurementGenBinomial,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty}}
    #tests={"knapsack":{"T":30,"reps":2,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorKnapsack,"dec_fct":DecisionLogicEmpty}}
    tests={"aspiration":{"T":100,"reps":2,"params":{"N":[5,10,20],"n1":[0],"n2":[2]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicAspiration}}
    #tests={"binomial":{"T":10,"reps":2,"params":{"N":10,"mu1":[1],"mu2":[5,20,50],"rich":[0.2,0.5,0.8]},"meas_fct":MeasurementGenBinomial,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty}}
    for test,conf in tests.items():
        run_experiment(test,conf)
