import functools
import itertools
import pandas as pd
import numpy as np
from src.Supervisor import BaseSupervisor
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
            model=BaseSupervisor(N=int(params["N"]),measurement_fct=f,
                                 bidsplit=conf['bidsplit'],multibid=conf['multibid'],
                                 decision_fct=conf["dec_fct"],
                                 agent_decision_fct=conf["dec_fct_agent"],
                                 reward_fct=conf["rew_fct"],
                                 evaluation_fct=conf["eval_fct"],
                                 agent_type=NegoAgent)
            model.run(params=params)
            ## save intermediate results
            log_tot=log_tot+[model.get_log(params)]
        save_result(datadir,test,{k:v for k,v in params.items() if k!="repetition"},log_tot[-int(conf["reps"]):]) # log the last conf['reps'] iterations (repetitions)
    varnames=[k for k,v in conf["params"].items() if len(v)>1] # keep vars for which there is more than one value
    for varname in varnames:
        print("logging "+varname)
        stats_rew=get_stats(log_tot,"reward",idx=[varname])
        stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["production","consumption","tariff"])
        stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["action","cost"])
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
        plot_trend(stats_rew,varname,os.path.join(plotdir,"./rewards_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=[t for t in ["reward","produce_avg"] if t !=varname])
        plot_trend(stats_perc,varname,os.path.join(plotdir,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["production","consumption","tariff"])
        plot_trend(stats_decs,varname,os.path.join(plotdir,"./decisions_"+str(test)+"_"+str(varname)+"_nego.pdf"),trends=["action","cost"])
        plot_measures_nego(stats_eval,varname,os.path.join(plotdir,"./eval_"+str(test)+"_"+str(varname)+"_nego.pdf"))


if __name__ == '__main__':

    default_params={"reps":30,"dec_fct":NegoDecisionLogic,"dec_fct_agent":BaseDecisionLogic,"rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,"meas_fct":MeasurementGenReal,
                    "params":{"T":[10],"discrimination":[0.5], #TODO used??
                              "min_income":[20000],"max_income":[100000], # used to produce the income values
                              # "chance_rich":[0.2],"chance_poor":[0.5],    # proportion of rich and poor in each class, TODO used??
                              "mu1":[5.0],"mu2":[2.0],                    # determine average consumption
                              "produce_avg":[1],                          # determines average production
                              "buy_low":[0.25],"buy_high":[0.48],         # proportion of productive individuals is low and high caste
                              "N":[20,30,50,70,100],                      # population size
                              "low_caste":[0.36], # proportion of low caste agents?
                              "tariff_avg":[1],    # mean of tariff value, which determines the partner matching
                              "bias_low":[0.0],"bias_high":[0.0], # bias for agents
                              "bias_degree":[0.0]                 # bias for mediator
                    }}
    test_N={             # bid splitting should increase efficiency
        "test_N":{**default_params,"bidsplit":False,"multibid":False,
                           'params':{**default_params['params'],
                                     "N":[20,30,50,70,100,150],"bias_low":[0.0],"bias_high":[0.0],"bias_degree":[0.0]}}}
    test_production={           # higher production should reduce efficiency if bid splitting is disabled
        "test_prod_nosplit":{**default_params,"bidsplit":False,"multibid":False,
                           'params':{**default_params['params'],
                                     "N":[20,30,50,70,100,150],"produce_avg":[1.0,2.0,3.0,5,0]}},
        "test_prod_split":{**default_params,"bidsplit":True,"multibid":False,
                           'params':{**default_params['params'],
                                     "N":[20,30,50,70,100,150],"produce_avg":[1.0,2.0,3.0,5,0]}}}
    test_bidsplit={             # bid splitting should increase efficiency
        "test_nobidsplit":{**default_params,"bidsplit":False,"multibid":False,
                           'params':{**default_params['params'],
                                     "N":[20,30,50,70,100,150],"bias_low":[0.0],"bias_high":[0.0],"bias_degree":[0.0]}}, # no bias
        "test_bidsplit":{**default_params,"bidsplit":True,"multibid":False,
                         'params':{**default_params['params'],
                                   "N":[20,30,50,70,100,150],"bias_low":[0.0],"bias_high":[0.0],"bias_degree":[0.0]}}} # no bias
    test_bias={          # bias should reduce efficiency and market access
        "test_bias_agents":{**default_params,"bidsplit":False,"multibid":False,
                           'params':{**default_params['params'],
                                     "N":[20,30,50,70,100,150],"bias_low":[0.0,0.2,0.5],"bias_high":[0.0,0.2,0.5],"bias_degree":[0.0]}}, # no mediator bias
        "test_bias_mediator":{**default_params,"bidsplit":False,"multibid":False,
                         'params':{**default_params['params'],
                                   "N":[20,30,50,70,100,150],"bias_low":[0.0],"bias_high":[0.0],"bias_degree":[0.0,0.2,0.5]}}} # no biased agents
    experiments={"exp_base":{**default_params,"bidsplit":False,"multibid":False,
                             'params':{**default_params['params'],
                                       "bias_low":[0.0],"bias_high":[0.2,0.5,0.8], # agents are biased
                                       "bias_degree":[0.0] # Mediator is not biased
                                       }},
                 "exp1":{**default_params,"bidsplit":False,"multibid":False,
                             'params':{**default_params['params'],
                                       "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                       "bias_degree":[0.0] # Mediator is not biased
                                       }},
                 "exp2":{**default_params,"bidsplit":True,"multibid":False, # bid splitting
                             'params':{**default_params['params'],
                                       "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                       "bias_degree":[0.0] # Mediator is not biased
                                       }},
                 "exp3":{**default_params,"bidsplit":False,"multibid":False, # no bid splitting
                             'params':{**default_params['params'],
                                       "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                       "bias_degree":[0.1,0.5,1.0] # Mediator is biased
                                       }},
                 "exp4":{**default_params,"bidsplit":True,"multibid":False, # bid splitting
                             'params':{**default_params['params'],
                                       "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                       "bias_degree":[0.1,0.5,1.0] # Mediator is biased
                                       }}}

    for test,conf in test_production.items():
        run_experiment(test,conf,"./output")
