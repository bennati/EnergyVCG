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
from nego.src.utilsnego import *
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
        # stats_rew=get_stats(log_tot,"reward",idx=[varname])
        # stats_perc=get_stats(log_tot,"perception",idx=[varname],cols=["production","consumption","tariff"])
        # stats_decs=get_stats(log_tot,"decisions",idx=[varname],cols=["action","cost"])
        stats_eval=get_stats(log_tot,"evaluation",idx=[varname],cols=["gini","efficiency",
                                                                      "wealth_distribution",
                                                                      "social_welfare",
                                                                      "market_access"])
        #"wealth_distribution_high","wealth_distribution_low",#"social_welfare_high","social_welfare_low","market_access_high","market_access_low"])
        # nn = (pd.DataFrame([log_tot[i]["evaluation"]["efficiency"] for i in range(conf["reps"])])[0]).tolist()
        # nn.append("efficiency")
        # rr = (pd.DataFrame([log_tot[i]["evaluation"]["gini"] for i in range(conf["reps"])])[0]).tolist()
        # rr.append("gini")
        # q = (pd.DataFrame([log_tot[i]["evaluation"]["market_access"] for i in range(conf["reps"])])[0]).tolist()
        # q.append("market_access")
        # r = (pd.DataFrame([log_tot[i]["evaluation"]["market_access_high"] for i in range(conf["reps"])])[0]).tolist()
        # r.append("market_access_high")
        # p = (pd.DataFrame([log_tot[i]["evaluation"]["market_access_low"] for i in range(conf["reps"])])[0]).tolist()
        # p.append("market_access_low")
        # m = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare"] for i in range(conf["reps"])])[0]).tolist()
        # m.append("social_welfare")
        # n = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare_high"] for i in range(conf["reps"])])[0]).tolist()
        # n.append("social_welfare_high")
        # o = (pd.DataFrame([log_tot[i]["evaluation"]["social_welfare_low"] for i in range(conf["reps"])])[0]).tolist()
        # o.append("social_welfare_low")
        # tt = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution"] for i in range(conf["reps"])])[0]).tolist()
        # tt.append("wealth_distribution")
        # pp = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution_high"] for i in range(conf["reps"])])[0]).tolist()
        # pp.append("wealth_distribution_high")
        # qq = (pd.DataFrame([log_tot[i]["evaluation"]["wealth_distribution_low"] for i in range(conf["reps"])])[0]).tolist()
        # qq.append("wealth_distribution_low")
        # with open("log_"+str(varname)+".csv",'w') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # wr.writerow(nn)
            # wr.writerow(rr)
            # wr.writerow(q)
            # wr.writerow(r)
            # wr.writerow(p)
            # wr.writerow(m)
            # wr.writerow(n)
            # wr.writerow(o)
            # wr.writerow(tt)
            # wr.writerow(pp)
            # wr.writerow(qq)
        stats_eval.to_csv(os.path.join(datadir,test,"evaluations_"+str(varname)+".csv"))
        # plot_trend(stats_rew,varname,"./rewards_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_trend(stats_perc,varname,"./perceptions_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_trend(stats_decs,varname,"./decisions_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_measures(stats_eval,varname,"./eval_"+str(test)+"_"+str(varname)+"_nego.png")
        # plot_measures1(stats_eval1,varname,"./eval_1_"+str(test)+"_"+str(varname)+"_nego.png")


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
    # test_N={"test_N":{**default_params,'params':{**default_params['params'],
    #                                              "N":[20,30,50,70,100],"mu1":[1.01],
    #                                              "low_caste":[0.36],"buy_low":[0.25],
    #                                              "bias_low":[0.02],"bias_high":[0.8],"bias_degree":[0.5],
    #                                              "tariff_avg":[1],"produce_avg":[1]}}}
    # test_low_caste={"test_low_caste":{**default_params,
    #                                   'params':{**default_params['params'],
    #                                             "N":[50],"mu1":[1.01],
    #                                             "low_caste":[0.2,0.36,0.8],"buy_low":[0.25],
    #                                             "bias_low":[0.02],"bias_high":[0.8],"bias_degree":[0.5],
    #                                             "tariff_avg":[1],"produce_avg":[1]}}}
    # test_buy_low={"test_buy_low":{**default_params,
    #                               'params':{**default_params['params'],
    #                                         "N":[50],"mu1":[1.01],
    #                                         "low_caste":[0.36],"buy_low":[0.25,0.5,0.8],
    #                                         "bias_low":[0.02],"bias_high":[0.8],"bias_degree":[0.5],
    #                                         "tariff_avg":[1],"produce_avg":[1]}}}
    # test_bias_high={"test_bias_high":{**default_params,
    #                                   'params':{**default_params['params'],
    #                                             "N":[50],"mu1":[1.01],
    #                                             "low_caste":[0.36],"buy_low":[0.25],
    #                                             "bias_low":[0.02],"bias_high":[0.2,0.5,0.8],"bias_degree":[0.5],
    #                                             "tariff_avg":[1],"produce_avg":[1]}}}
    # test_bias_degree={"test_bias_degree":{**default_params,
    #                                       'params':{**default_params['params'],
    #                                                 "N":[50],"mu1":[1.01],
    #                                                 "low_caste":[0.36],"buy_low":[0.25],
    #                                                 "bias_low":[0.02],"bias_high":[0.5],"bias_degree":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #                                                 "tariff_avg":[1],"produce_avg":[1]}}}
    # test_consumption={"test_consumption":{**default_params,
    #                                       'params':{**default_params['params'],
    #                                                 "N":[50],"mu1":[1.01,2.02,3.03],
    #                                                 "low_caste":[0.36],"buy_low":[0.25],
    #                                                 "bias_low":[0.5],"bias_high":[0.5],"bias_degree":[0.5],
    #                                                 "tariff_avg":[1],"produce_avg":[1]}}}
    # test_tariff={"test_tariff":{**default_params,'params':{**default_params['params'],
    #                                                        "N":[50],"mu1":[1.01],
    #                                                        "low_caste":[0.36],"buy_low":[0.25],
    #                                                        "bias_low":[0.25],"bias_high":[0.5],"bias_degree":[0.5],
    #                                                        "tariff_avg":[1,2,3],"produce_avg":[1]}}}
    # test_production={"test_production":{**default_params,
    #                                     'params':{**default_params['params'],
    #                                               "N":[50],"mu1":[1.01],
    #                                               "low_caste":[0.36],"buy_low":[0.25],
    #                                               "bias_low":[0.5],"bias_high":[0.5],"bias_degree":[0.5],
    #                                               "tariff_avg":[1],"produce_avg":[1,2,3]}}}
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
