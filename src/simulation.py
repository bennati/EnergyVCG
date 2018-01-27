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

def run_experiment(test,conf):
    log_tot=[]
    for r in range(conf["reps"]):
        print("repetition: "+str(r))
        for idx,p in expandgrid(conf["params"]).iterrows():
            params=p.to_dict()
            params.update({"repetition":r})
        f=functools.partial(conf["meas_fct"],**params)
            model=BaseSupervisor(params["N"],measurement_fct=f,decision_fct=conf["dec_fct_sup"],agent_decision_fct=conf["dec_fct"],reward_fct=conf["rew_fct"],agent_type=BaseAgent)
            model.run(conf["T"],params=params)
        log_tot=log_tot+model.log # concatenate lists
    # compute statistics for all tables in log file
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        stats_rew=get_stats(log_tot,"reward",idx=[varname],cols=["reward"])
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["value","cost"])
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["contribution","cost","contributed"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","cost","efficiency","social_welfare","success","num_contrib"])
        plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+".pdf")
        plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+".pdf")
        plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+".pdf")


if __name__ == '__main__':
    # tests={"uniform":{"T":10,"reps":2,"params":{"N":[10],"mu":[5,20,50]},"meas_fct":MeasurementGenNormal,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform},"binomial":{"T":10,"reps":2,"params":{"N":[10],"mu1":[1],"mu2":[5,20,50],"rich":[0.2,0.5,0.8]},"meas_fct":MeasurementGenBinomial,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    tests={"knapsack":{"T":50,"reps":10,"params":{"N":[10,20,30],"n1":[0],"n2":[2,5,8]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorKnapsack,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    #tests={"aspiration":{"T":100,"reps":2,"params":{"N":[5,10,20],"n1":[0],"n2":[2]},"meas_fct":MeasurementGenUniform,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicAspiration}}
    # tests={"qlearn":{"T":10,"reps":2,"params":{"N":[5,10],"n1":[1],"n2":[2,5]},"meas_fct":MeasurementGenQlearn,"dec_fct_sup":DecisionLogicSupervisorEmpty,"dec_fct":DecisionLogicQlearn,"rew_fct":RewardLogicQlearn}}
    #tests={"binomial":{"T":10,"reps":2,"params":{"N":10,"mu1":[1],"mu2":[5,20,50],"rich":[0.2,0.5,0.8]},"meas_fct":MeasurementGenBinomial,"dec_fct_sup":DecisionLogicSupervisorMandatory,"dec_fct":DecisionLogicEmpty}}
    for test,conf in tests.items():
        run_experiment(test,conf)
    test,conf=list(tests.items())[0]
