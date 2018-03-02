import functools
import itertools
import pandas as pd
import numpy as np
from copy import copy
import functools
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
    ret=ret.transform(lambda x: [[k,v/sum([b for a,b in x])] for k,v in x]) # normalize by population size
    ## for each parameter value returns a dataframe where the index (level 0, level1 is a counter) is N (the number of times an agent contributes) and each value counts the number of agents that contributed N times in a given repetition (it lists the outcomes of all repetitions, disaggregated, so that they can be aggregated with other parameter settings later on)
    ret=pd.DataFrame(ret).groupby(varnames,as_index=True).apply(lambda x: pd.DataFrame(np.concatenate(np.asarray(x[0])),columns=["value","cnt"]).groupby(["value"],as_index=True).apply(lambda x: pd.DataFrame(pd.concat([x["cnt"]]))))
    ret=ret.reset_index(level=[0,1,2]).reset_index(level=0,drop=True) # convert indexes to columns
    return ret

def run_experiment(test,conf):
    print("starting "+str(test))
    if not os.path.exists("data/"+str(test)):
        os.makedirs("data/"+str(test))
    log_tot=[]
    qtables=[]
    qlearning=False
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
                tab2=pd.concat([a.decision_fct.get_qcount().assign(idx=a.unique_id) for a in model.schedule.agents]) # get the number of experiences for each state
                tab["index"]=tab.index
                tab2["index"]=tab2.index
                tab=pd.merge(tab,tab2,right_on=["idx","index"],left_on=["idx","index"]) # merge along the index and agent id
                tab["repetition"]=r
                print(params)
                print(tab)
                for k,v in dict(p).items():
                    tab[k]=v
                qtables=qtables+[tab]
                qlearning=True
            except:
                print("Qtable not defined")
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    ## save results
    res_decs=pd.concat([pd.DataFrame(i["decisions"]) for i in log_tot])
    res_decs.to_csv("./data/"+str(test)+"/decisions.csv.gz",index=False,compression='gzip')
    res_percs=pd.concat([pd.DataFrame(i["perception"]) for i in log_tot])
    res_percs.to_csv("./data/"+str(test)+"/perceptions.csv.gz",index=False,compression='gzip')
    res_rew=pd.concat([pd.DataFrame(i["reward"]) for i in log_tot])
    res_rew.to_csv("./data/"+str(test)+"/rewards.csv.gz",index=False,compression='gzip')
    res_eval=pd.concat([pd.DataFrame(i["evaluation"]) for i in log_tot])
    res_eval.to_csv("./data/"+str(test)+"/evaluation.csv.gz",index=False,compression='gzip')
    ### prepare tables ###
    if varnames:
        contrib_hist=compute_contrib_hist(res_decs,varnames)
    contrib_hist.to_csv("./data/"+str(test)+"/contrib_hist.csv.gz",index=False,compression='gzip')
    #contrib_hist=pd.read_csv("./data/contrib_hist.csv.gz")
    # aggregate over all agent ids and compute gini coefficients
        stats_gini_contribs=res_decs.groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum}) # sum up all contributions in each simulation (over all timesteps)
    stats_gini_contribs=stats_gini_contribs.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini}) # compute gini coefficient across agents
    stats_gini_contribs=stats_gini_contribs.rename(columns={"contributed":"Contributors","contribution":"Values"})
    stats_gini_contribs.to_csv("./data/"+str(test)+"/stats_gini_contribs.csv.gz",index=False,compression='gzip')
        stats_contrib_hist2=compute_stats(contrib_hist,idx=varnames+["value"],columns=["cnt"])
    #stats_gini_contribs=pd.read_csv("./data/"+str(test)+"/stats_gini_contribs.csv.gz")
    ## compute stats qtables
    if qlearning:
        qtables=pd.concat(qtables)
         # transform the index to two separate columns
        qtables["state_val"]=qtables["index"].transform(lambda x: x[0])
        qtables["state_cost"]=qtables["index"].transform(lambda x: x[1])
        qtables.drop("index",axis=1,inplace=True)
        qtables["prob"]=[boltzmann([r[1]["yes"],r[1]["no"]],0.1)[0] for r in qtables.iterrows()] # normalize qvalues, prob is the probability of contributing using the boltzmann equation
        qtables.to_csv("./data/"+str(test)+"/qtables.csv.gz",index=False,compression='gzip')

if __name__ == '__main__':
    tests={"qlearn":{"T":1000,"reps":3,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicQlearn,"rew_fct":RewardLogicUniform}}#
    #tests={"aspiration":{"T":100,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,3,4,5,6,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicAspiration,"rew_fct":RewardLogicUniformAspir}}
    # tests={"mandatory":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    # tests={"knapsack":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorKnapsack,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    for test,conf in tests.items():
        run_experiment(test,conf)
    test,conf=list(tests.items())[0]
