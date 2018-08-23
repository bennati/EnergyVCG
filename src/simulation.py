from multiprocessing import Pool
import itertools
import pandas as pd
import numpy as np
from copy import copy
import functools
from DecisionLogic import BaseDecisionLogic
from RewardLogic import *
from MeasurementGen import *
from sim_default import *
from sim_nrel_data import *
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
    ret=ret.reset_index(level=list(range(len(varnames)+1))).reset_index(level=0,drop=True) # convert indexes to columns
    return ret

def compute_qtabs(n,p,model):
    ret=None
    losses=None
    try:
        tab,losses=model.decision_fct.get_qtable()
        # print(tab)
        tab=tab.rename(columns={0:"no",1:"yes"})
        # tab2=model.decision_fct.get_qcount() # get the number of experiences for each state
        tab["index"]=tab.index
        # tab2["index"]=tab2.index
        # tab=pd.merge(tab,tab2,right_on=["idx","index"],left_on=["idx","index"]) # merge along the index and agent id
        tab["repetition"]=n
        print(dict(p))
        for k,v in dict(p).items():
            tab[k]=v
        ret=tab
    except Exception as e:
        print("Qtable not defined")
        print(e)
    return ret,losses

# def body_q(n,conf):
#     tf.set_random_seed(n)
#     np.random.seed(n)
#     print("repetition: "+str(n))
#     log=[]
#     qtab=[]
#     for idx,p in expandgrid(conf["params"]).iterrows():
#         params=p.to_dict()
#         print(params)
#         params.update({"repetition":n,"T":conf["T"]})
#         f=functools.partial(conf["meas_fct"],**params)
#         model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=conf["rew_fct"],agent_type=BaseAgent)
#         model.run(params=params)
#         log=log+model.log
#         qtab=qtab+[compute_qtabs(n,p,model)]
#     return log,qtab

def body(n,conf,test,datadir,gamma=0.0,alpha=0.001):
    np.random.seed(n)
    tf.set_random_seed(np.random.uniform(n))
    print("repetition: "+str(n))
    res_decs=pd.DataFrame()
    for idx,p in expandgrid(conf["params"]).iterrows():
        params=p.to_dict()
        print(params)
        params.update({"repetition":n,"T":conf["T"]})
        f=functools.partial(conf["meas_fct"],**params)
        model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=conf["rew_fct"],agent_type=BaseAgent,alpha=conf["A"],gamma=conf["G"])
        model.run(params=params)
        res_decs=pd.concat([res_decs,save_result(datadir,test,params,model.log)])
        save_qtab(datadir,test,compute_qtabs(n,p,model),params)
    return res_decs

def save_result(datadir,test,params,log):
    res_decs=pd.concat([pd.DataFrame(i["decisions"]) for i in log])
    res_decs.to_csv(os.path.join(datadir,str(test),"decisions_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')
    for l in ["perception","reward","evaluation"]:
        pd.concat([pd.DataFrame(i[l]) for i in log]).to_csv(os.path.join(datadir,str(test),str(l)+"_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')
    return res_decs

# def save_qtabs(test,qtables):
#     if any([(q is not None) and (not q.empty) for q in qtables]):            # if at least a table is not None
#         ## save to file
#         qtables=pd.concat(qtables)
#          # transform the index to two separate columns
#         qtables["state_val"]=qtables["index"].transform(lambda x: x[0])
#         qtables["state_cost"]=qtables["index"].transform(lambda x: x[1])
#         qtables.drop("index",axis=1,inplace=True)
#         if all([i in qtables.columns for i in ["yes","no"]]):
#             qtables["prob"]=[boltzmann([r[1]["yes"],r[1]["no"]],0.1)[0] for r in qtables.iterrows()] # normalize qvalues, prob is the probability of contributing using the boltzmann equation
#         qtables.to_csv("./data/"+str(test)+"/qtables.csv.gz",index=False,compression='gzip')
#         del qtables

def save_qtab(datadir,test,qtabs,params):
    qtab=qtabs[0]
    losses=qtabs[1]
    if qtab is not None and not qtab.empty:
         # transform the index to two separate columns
        qtab["state_val"]=qtab["index"].transform(lambda x: x[0])
        qtab["state_cost"]=qtab["index"].transform(lambda x: x[1])
        qtab.drop("index",axis=1,inplace=True)
        if all([i in qtab.columns for i in ["yes","no"]]):
            qtab["prob"]=[boltzmann([r[1]["yes"],r[1]["no"]],0.1)[0] for r in qtab.iterrows()] # normalize qvalues, prob is the probability of contributing using the boltzmann equation
        qtab.to_csv(os.path.join(datadir,str(test),"qtab_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')
    if losses is not None and not losses.empty:
        losses.to_csv(os.path.join(datadir,str(test),"loss_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')

def save_stats(datadir,test,conf,res_decs,time_min=3000):
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    ### prepare tables ###
    if varnames:
        contrib_hist=compute_contrib_hist(res_decs,varnames)
        contrib_hist.to_csv(os.path.join(datadir,str(test),"contrib_hist.csv.gz"),index=False,compression='gzip')
        #contrib_hist=pd.read_csv(os.path.join(datadir,"contrib_hist.csv.gz"))
        # stats_contrib_hist2=compute_stats(contrib_hist,idx=varnames+["value"],columns=["cnt"])
        del contrib_hist
        # aggregate over all agent ids and compute gini coefficients
        stats_gini_contribs=res_decs[res_decs["timestep"]>time_min].groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum,"cost":np.sum}) # sum up all contributions in each simulation (over all timesteps)
        stats_gini_contribs=stats_gini_contribs.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini,"cost":gini}) # compute gini coefficient across agents
        stats_gini_contribs=stats_gini_contribs.rename(columns={"contributed":"Contributors","contribution":"Values"})
        stats_gini_contribs.to_csv(os.path.join(datadir,str(test),"stats_gini_contribs.csv.gz"),index=False,compression='gzip')
        del stats_gini_contribs
    #stats_gini_contribs=pd.read_csv(os.path.join(datadir,str(test),"stats_gini_contribs.csv.gz"))

def run_experiment_par(test,conf,datadir):
    print("starting "+str(test))
    if not os.path.exists(os.path.join(datadir,test)):
        os.makedirs(os.path.join(datadir,test))
    # empty dir
    for files in os.listdir(os.path.join(datadir,test)):
        f=os.path.join(datadir,str(test),files)
        if os.path.isfile(f):
            os.unlink(f)
    part_fun=functools.partial(body,conf=conf,test=test,datadir=datadir,alpha=conf["A"],gamma=conf["G"])
    if __name__ == '__main__':
        print("starting processes")
        pool=Pool()
        ans=pool.map(part_fun,range(conf["reps"]))
    else:
        ans=map(part_fun,range(conf["reps"]))
    # save_stats(datadir,test,conf,pd.concat(ans))

def run_experiment(test,conf,datadir):
    print("starting "+str(test))
    if not os.path.exists(os.path.join(datadir,test)):
        os.makedirs(os.path.join(datadir,test))
    # empty dir
    for files in os.listdir(os.path.join(datadir,test)):
        f=os.path.join(datadir,str(test),files)
        if os.path.isfile(f):
            os.unlink(f)
    # log_tot=[]
    # qtab_list=[]
    res_decs=pd.DataFrame()
    for r in range(conf["reps"]):
        print("repetition: "+str(r))
        for idx,p in expandgrid(conf["params"]).iterrows():
            params=p.to_dict()
            print(params)
            params.update({"repetition":r,"T":conf["T"]})
            f=functools.partial(conf["meas_fct"],**params)
            model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=conf["rew_fct"],agent_type=BaseAgent,alpha=conf["A"],gamma=conf["G"])
            model.run(params=params)
            ## save intermediate results
            res_decs=pd.concat([res_decs,save_result(datadir,test,params,model.log)])
            ## compute qtables
            save_qtab(datadir,test,compute_qtabs(r,p,model),params)
    # compute statistics for all tables in log file
    # save_stats(datadir,test,conf,res_decs)
