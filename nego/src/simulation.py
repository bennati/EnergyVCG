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

def save_result(df,datadir,test,params,l):
    """
    Log is a list that is 'reps' long, which contains lists 'T' long, which contain dictionaries
    """
    df.to_csv(os.path.join(datadir,str(test),str(l)+"_"+str("_".join([str(k)+str(v) for k,v in params.items()]))+".csv"),index=False)

def run_experiment(test,conf,datadir="./"):
    print("starting "+str(test))
    if not os.path.exists(os.path.join(datadir,test)):
        os.makedirs(os.path.join(datadir,test))
    res_rew=[]
    res_perc=[]
    res_eval=[]
    res_decs=[]
    for idx,p in expandgrid(conf["params"]).iterrows():
        print("params "+str(p))
        log_tot=[]
        for r in range(conf["reps"]):
            params=p.to_dict()
            params.update({"repetition":r})
            fm=functools.partial(conf["meas_fct"],**params)
            fd=functools.partial(conf["dec_fct"],bias_fct=conf['bias_fct'])
            model=NegoSupervisor(N=int(params["N"]),measurement_fct=fm,
                                 bidsplit=conf['bidsplit'],multibid=conf['multibid'],
                                 decision_fct=fd,
                                 agent_decision_fct=conf["dec_fct_agent"],
                                 reward_fct=conf["rew_fct"],
                                 evaluation_fct=conf["eval_fct"],
                                 agent_type=NegoAgent)
            model.run(params=params)
            log=model.log
            ## save intermediate results
            log_tot=log_tot+[log]
        ## Log is a list that is 'reps' long, which contains lists 'T' long, which contain dictionaries
        res_rew.append(pd.concat([pd.concat([pd.DataFrame(t["reward"]) for t in reps]) for reps in log_tot]))
        save_result(res_rew[-1],datadir,test,p.to_dict(),"reward")
        res_perc.append(pd.concat([pd.concat([pd.DataFrame(t["perception"]) for t in reps]) for reps in log_tot]))
        save_result(res_perc[-1],datadir,test,p.to_dict(),"perception")
        res_eval.append(pd.concat([pd.concat([pd.DataFrame(t["evaluation"]) for t in reps]) for reps in log_tot]))
        save_result(res_eval[-1],datadir,test,p.to_dict(),"evaluation")
        res_decs.append(pd.concat([pd.concat([pd.DataFrame(t["decisions"]) for t in reps]) for reps in log_tot]))
        save_result(res_decs[-1],datadir,test,p.to_dict(),"decisions")
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        print("logging "+varname)
        stats_rew=compute_stats(res_rew,idx=[varname])
        stats_perc=compute_stats(res_perc,idx=[varname],columns=["production","consumption","tariff"])
        # stats_decs=compute_stats(res_decs,idx=[varname],columns=["cost"])
        stats_eval=compute_stats(res_eval,idx=[varname],columns=["market_access",
                                                                      "market_access_low",
                                                                      "market_access_high",
                                                                 "trade_low_low",
                                                                 "trade_high_low",
                                                                 "trade_low_high",
                                                                 "trade_high_high",
                                                                      "sum_surplus_prod_low",
                                                                      "sum_surplus_prod_high",
                                                                      "sum_surplus_cons_low",
                                                                      "sum_surplus_cons_high",
                                                                 "sum_initial_prod_low",
                                                                 "sum_initial_prod_high",
                                                                 "sum_initial_cons_low",
                                                                 "sum_initial_cons_high",
                                                                 "satifaction_cons_low",
                                                                 "satifaction_cons_high",
                                                                 "satifaction_prod_low",
                                                                 "satifaction_prod_high",
                                                                 "efficiency",
                                                                 "inequality"])
        stats_rew.to_csv(os.path.join(datadir,test,"agg_rewards_"+str(varname)+".csv"),index=False)
        stats_perc.to_csv(os.path.join(datadir,test,"agg_perceptions_"+str(varname)+".csv"),index=False)
        # stats_decs.to_csv(os.path.join(datadir,test,"agg_decisions_"+str(varname)+".csv"),index=False)
        stats_eval.to_csv(os.path.join(datadir,test,"agg_evaluations_"+str(varname)+".csv"),index=False)
        plotdir=os.path.join(datadir,test,"plots")
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        plot_trend(stats_rew,varname,os.path.join(plotdir,"./rewards_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=[t for t in ["reward"] if t !=varname])
        plot_trend(stats_perc,varname,os.path.join(plotdir,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["production","consumption","tariff"])
        # plot_trend(stats_decs,varname,os.path.join(plotdir,"./decisions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["cost"])
        plot_measures_nego(stats_eval,varname,os.path.join(plotdir,"./eval_"+str(test)+"_"+str(varname)+"_nego.pdf"))
