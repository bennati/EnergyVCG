import functools
import itertools
import pandas as pd
import numpy as np
from copy import copy
from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from MeasurementGen import BaseMeasurementGen
from sim_default import *
from sim_qlearn import *
from sim_aspiration import *
from sim_knapsack import *
from utils import *
from Agent import *
from Supervisor import *

def compute_contrib_hist(decisions,varnames):
    """
    Decisions: a list of dataframes containing the decisions
    varnames: the names of the columns defining the parameters of the simulation

    Returns:
    A dataframe containing the number of agents (cnt) contributing N times (value) in a simulation, for each scenario (varnames)
    The dataframe might contain more than one record for each value, one for each repetion in which that value exists.
    This allows to aggregate the result over different parameters

    The dataframe has the following columns:
    value: is the number of contributions an agent has done in the simulation e.g. it contributed N timesteps of a total of 50 timesteps
    cnt: is the number of agents that contributed N during one simulations e.g. 3 agents contributed a total of N times (they can be different timesteps)
    """
    ret=decisions.copy().groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum}) # sum up all contributions in each simulation (over all timesteps)
    ## count how many times each agent contributed: value_counts returns the number of times each N occurs
    ret=ret.groupby(varnames+["repetition"],as_index=False).apply(lambda x: np.asarray(pd.DataFrame(x["contributed"].value_counts()).reset_index()))
    ## for each parameter value returns a dataframe where the index (level 0, level1 is a counter) is N (the number of times an agent contributes) and each value counts the number of agents that contributed N times in a given repetition (it lists the outcomes of all repetitions, disaggregated, so that they can be aggregated with other parameter settings later on)
    ret=pd.DataFrame(ret).groupby(varnames,as_index=True).apply(lambda x: pd.DataFrame(np.concatenate(np.asarray(x[0])),columns=["value","count"]).groupby(["value"],as_index=True).apply(lambda x: pd.concat([x["count"]])))
    ret=ret.reset_index(level=[0,1,2]).reset_index(level=0,drop=True) # convert indexes to columns
    ret=ret.rename(columns={"count":"cnt"}) # rename to avoid conflicts later with compute_stats
    return ret

def run_experiment(test,conf):
    log_tot=[]
    qtables=[]
    for r in range(conf["reps"]):
        print("repetition: "+str(r))
        for idx,p in expandgrid(conf["params"]).iterrows():
            params=p.to_dict()
            params.update({"repetition":r})
        f=functools.partial(conf["meas_fct"],**params)
            model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=conf["rew_fct"],agent_type=BaseAgent)
            model.run(conf["T"],params=params)
        log_tot=log_tot+model.log # concatenate lists
            try:
                tab=pd.concat([a.decision_fct.get_qtable().assign(idx=a.unique_id) for a in model.schedule.agents])
                tab=tab.rename(columns={0:"no",1:"yes"})
                tab["repetition"]=r
                for k,v in dict(p).items():
                    tab[k]=v
                qtables=qtables+[tab]
            except:
                print("Qtable not defined")
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    ## TODO count how many times they enter each state
    ## TODO each state/action around 10 steps
    # compute stats qtables
    qtables=compute_stats([pd.concat(qtables).reset_index()],idx=varnames+["index"],columns=["no","yes"])
    stats_evalt=get_stats(log_tot,"evaluation",idx=["timestep"],cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
    plot_measures(stats_evalt,"timestep","./eval_"+str(test)+"_"+str("time")+".pdf")
    stats_gini_contribs=pd.concat([pd.DataFrame(i["decisions"]) for i in log_tot])
    contrib_hist=compute_contrib_hist(stats_gini_contribs,varnames)
    ## aggregate over all agent ids and compute gini coefficients
    stats_gini_contribs=stats_gini_contribs.groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum}) # sum up all contributions in each simulation (over all timesteps)
    stats_gini_contribs=stats_gini_contribs.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini}) # compute gini coefficient across agents
    stats_gini_contribs=stats_gini_contribs.rename(columns={"contributed":"Contributors","contribution":"Values"})
    for varname in varnames:
        stats_gini=compute_stats([stats_gini_contribs],[varname],columns=["Contributors","Values"]) # average across repetitions
        plot_trend(stats_gini,varname,"./gini_"+str(test)+"_"+str(varname)+".pdf")
        # idx_lvl=np.where(np.asarray(contrib_hist.index.names)==varname)[0][0]
        # idx_vals=contrib_hist.index.levels[idx_lvl]
        stats_rew=get_stats(log_tot,"reward",idx=[varname],cols=["reward"])
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["value","cost"])
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["contribution","cost","contributed"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
        stats_contrib_hist=compute_stats(contrib_hist,idx=[varname,"value"],columns=["cnt"])
        plot_trend(stats_contrib_hist,"value","./contrib_hist_"+str(test)+"_"+str(varname)+".pdf",yname=varname)
        plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+".pdf")
        plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+".pdf")


if __name__ == '__main__':
    tests={"qlearn":{"T":500,"reps":5,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicQlearn,"rew_fct":RewardLogicUniform}}#{"aspiration":{"T":20,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,4,5,6,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicAspiration,"rew_fct":RewardLogicUniform},"mandatory":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform},"knapsack":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorKnapsack,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    for test,conf in tests.items():
        run_experiment(test,conf)
    test,conf=list(tests.items())[0]
