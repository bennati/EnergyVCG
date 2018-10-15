import functools
import itertools
import pandas as pd
import numpy as np
from nego.src.Supervisor import NegoSupervisor
from src.DecisionLogic import BaseDecisionLogic
from nego.src.DecisionLogic import NegoDecisionLogic
from nego.src.RewardLogic import NegoRewardLogic
from nego.src.MeasurementGen import *
from nego.src.Evaluation import  NegoEvaluationLogic
from src.utils import *
from nego.src.Agent import *

def save_result(datadir,test,params,log):
    res_decs=pd.concat([pd.DataFrame(i["decisions"]) for i in log])
    res_decs.to_csv(os.path.join(datadir,str(test),"decisions_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')
    for l in ["perception","reward","evaluation"]:
        pd.concat([pd.DataFrame(i[l]) for i in log]).to_csv(os.path.join(datadir,str(test),str(l)+"_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv.gz"),index=False,compression='gzip')
    return res_decs

def run_experiment(test,conf,datadir="./"):
    print("starting "+str(test))
    if not os.path.exists(os.path.join(datadir,test)):
        os.makedirs(os.path.join(datadir,test))
    log_tot=[]
    for idx,p in expandgrid(conf["params"]).iterrows():
        print("params "+str(p))
        for r in range(conf["reps"]):
            params=p.to_dict()
            params.update({"repetition":r})
            f=functools.partial(conf["meas_fct"],**params)
            model=NegoSupervisor(N=int(params["N"]),measurement_fct=f,
                                 bidsplit=conf['bidsplit'],multibid=conf['multibid'],
                                 decision_fct=conf["dec_fct"],
                                 agent_decision_fct=conf["dec_fct_agent"],
                                 reward_fct=conf["rew_fct"],
                                 evaluation_fct=conf["eval_fct"],
                                 agent_type=NegoAgent)
            model.run(params=params)
            log=model.log[-1]
            ## save intermediate results
            log_tot=log_tot+[log]
        save_result(datadir,test,{k:v for k,v in params.items() if k!="repetition"},log_tot[-int(conf["reps"]):]) # log the last conf['reps'] iterations (repetitions)
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        print("logging "+varname)
        stats_rew=get_stats(log_tot,"reward",idx=[varname])
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["production","consumption","tariff"])
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["cost"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","efficiency",
                                                                      "wealth_distribution",
                                                                      "social_welfare",
                                                                      "market_access"])
        stats_rew.to_csv(os.path.join(datadir,test,"agg_rewards_"+str(varname)+".csv"),index=False)
        stats_perc.to_csv(os.path.join(datadir,test,"agg_perceptions_"+str(varname)+".csv"),index=False)
        stats_decs.to_csv(os.path.join(datadir,test,"agg_decisions_"+str(varname)+".csv"),index=False)
        stats_eval.to_csv(os.path.join(datadir,test,"agg_evaluations_"+str(varname)+".csv"),index=False)
        plotdir=os.path.join(datadir,test,"plots")
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        plot_trend(stats_rew,varname,os.path.join(plotdir,"./rewards_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=[t for t in ["reward"] if t !=varname])
        plot_trend(stats_perc,varname,os.path.join(plotdir,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["production","consumption","tariff"])
        plot_trend(stats_decs,varname,os.path.join(plotdir,"./decisions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["cost"])
        plot_measures_nego(stats_eval,varname,os.path.join(plotdir,"./eval_"+str(test)+"_"+str(varname)+"_nego.pdf"))
