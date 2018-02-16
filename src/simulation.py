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

def subset_df(df,conditions):
    ret=df[functools.reduce(np.logical_and,[(df[k]==v) for k,v in zip(conditions.index,conditions)])] # select only a subset of the table matching the parameters
    return ret.reset_index()

def run_experiment(test,conf):
    print("starting "+str(test))
    if not os.path.exists("plots/"+str(test)):
        os.makedirs("plots/"+str(test))
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
    ### prepare tables ###
    stats_gini_contribs=pd.concat([pd.DataFrame(i["decisions"]) for i in log_tot])
    contrib_hist=compute_contrib_hist(stats_gini_contribs,varnames)
    contrib_hist.to_csv("./data/"+str(test)+"/contrib_hist.csv.gz",index=False,compression='gzip')
    #contrib_hist=pd.read_csv("./data/contrib_hist.csv.gz")
    # aggregate over all agent ids and compute gini coefficients
    stats_gini_contribs=stats_gini_contribs.groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum}) # sum up all contributions in each simulation (over all timesteps)
    stats_gini_contribs=stats_gini_contribs.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini}) # compute gini coefficient across agents
    stats_gini_contribs=stats_gini_contribs.rename(columns={"contributed":"Contributors","contribution":"Values"})
    stats_gini_contribs.to_csv("./data/"+str(test)+"/stats_gini_contribs.csv.gz",index=False,compression='gzip')
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
        #qtables=pd.read_csv("./data/"+str(test)+"/qtables.csv.gz")
    ### start with computing stats across all parameter configurations ###
    # compute evaluation over time
    stats_evalt=get_stats(log_tot,"evaluation",idx=["timestep"],cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
    stats_evalt.to_csv("./data/"+str(test)+"/stats_evalt.csv.gz",index=False,compression='gzip')
    #stats_evalt=pd.read_csv("./data/"+str(test)+"/stats_evalt.csv.gz")
    plot_measures(stats_evalt,"timestep","./plots/"+str(test)+"/eval_"+str("time")+".pdf")
    if qlearning:
        qtables_stats=compute_stats([qtables],idx=["state_val","state_cost"],columns=["prob","num"])
        plot_trend(qtables_stats,"state_cost","./plots/"+str(test)+"/qtables_cost.pdf",yname="state_val",trends=["prob"])
        plot_trend(qtables_stats,"state_val","./plots/"+str(test)+"/qtables_val.pdf",yname="state_cost",trends=["prob"])
    ### now move to computing statistics that aggregate on one of the parameters ###
    for varname in varnames:
        stats_gini=compute_stats([stats_gini_contribs],[varname],columns=["Contributors","Values"]) # average across repetitions
        plot_trend(stats_gini,varname,"./plots/"+str(test)+"/gini_"+str(varname)+".pdf")
        stats_rew=get_stats(log_tot,"reward",idx=[varname],cols=["reward"])
        plot_trend(stats_rew,varname,"./plots/"+str(test)+"/rewards_"+str(varname)+".pdf")
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["value","cost"])
        plot_trend(stats_perc,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf")
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["contribution","cost","contributed"])
        plot_trend(stats_decs,varname,"./plots/"+str(test)+"/decisions_"+str(varname)+".pdf")
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
        plot_measures(stats_eval,varname,"./plots/"+str(test)+"/eval"+str(varname)+".pdf")
        stats_contrib_hist=compute_stats(contrib_hist,idx=[varname,"value"],columns=["cnt"])
        plot_trend(stats_contrib_hist,"value","./plots/"+str(test)+"/contrib_hist_"+str(varname)+".pdf",yname=varname)

    ### now compute statistics for each parameter configuration, aggregating only on repetitions ###
    stats_contrib_hist2=compute_stats(contrib_hist,idx=varnames+["value"],columns=["cnt"])
    stats_t=get_stats(log_tot,"evaluation",idx=["timestep"]+varnames,cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
    for idx,p in expandgrid({k:conf["params"][k] for k in varnames}).iterrows():
        pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        # temporal evolution of measures
        tmp=subset_df(stats_t,p)
        plot_measures(tmp,"timestep","./plots/"+str(test)+"/time_"+pdesc+".pdf")
        # distribution of contributions
        # tmp=subset_df(stats_contrib_hist2,p)
        # plot_trend(tmp,"value","./plots/"+str(test)+"/contrib_hist_"+pdesc+".pdf")

    ## compute qtable heatmaps
    if qlearning:
        stats_q=compute_stats([qtables],idx=["state_val","state_cost"]+varnames,columns=["prob","num"])
        for idx,p in expandgrid({k:conf["params"][k] for k in varnames}).iterrows():
            pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
            tmp=subset_df(stats_q,p)
            f=lambda df,col: np.histogram2d(df["state_cost"],df["state_val"],weights=df[col+"_mean"],bins=[np.append(df["state_val"].unique(),[df["state_val"].max()+1]),np.append(df["state_cost"].unique(),[df["state_cost"].max()+1])])
            heatmap_choice,xlabs,ylabs=f(tmp,"prob")
            plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_choice"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
            # heatmap_count,xlabs,ylabs=f(tmp,"num")
            # plot_hmap(heatmap_count,"Average number of occurrences of a state",str(test)+"/heat_q_count"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
            ## compute histograms
            q_exp=subset_df(qtables,p) # subset with current experimental conditions
            plot_qtable_hist(q_exp,"./plots/"+str(test)+"/qhist_"+pdesc+".pdf","state_val","state_cost","prob",str(dict(p)))
            prefix="./plots/"+str(test)+"/heatmaps"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            plot_qtable_heat(q_exp,prefix+"/qheat_"+pdesc,"state_val","state_cost","prob")

if __name__ == '__main__':
    tests={"qlearn":{"T":1000,"reps":3,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicQlearn,"rew_fct":RewardLogicUniform}}#
    #tests={"aspiration":{"T":100,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,3,4,5,6,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicAspiration,"rew_fct":RewardLogicUniformAspir}}
    # tests={"mandatory":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    # tests={"knapsack":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorKnapsack,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    for test,conf in tests.items():
        run_experiment(test,conf)
    test,conf=list(tests.items())[0]
