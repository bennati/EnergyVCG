from src.Supervisor import BaseSupervisor
from src.DecisionLogic import BaseDecisionLogic
from nego.src.DecisionLogic import NegoDecisionLogic
from nego.src.RewardLogic import NegoRewardLogic
from nego.src.MeasurementGen import MeasurementGenReal
from nego.src.Evaluation import  NegoEvaluationLogic
from nego.src.simulation import *

Ns=[20,30,50,70,100]
biases=[0.0,0.2,0.5,0.8]

default_params={"reps":10,       # how many times to repeat the experiments
                "dec_fct":NegoDecisionLogic,"dec_fct_agent":BaseDecisionLogic,"rew_fct":NegoRewardLogic, "eval_fct":NegoEvaluationLogic,"meas_fct":MeasurementGenReal, # the logic to use during the experiments
                "params":{                           # the parameters of the experiments
                    "N":Ns,                          # population size
                    "low_caste":[None], # proportion of low caste agents, when None it uses the proportion in the data
                    "T":[10],                        # the duration of a run
                    "consumption_offset":[0.05], # offsets the consumption, as househols is the USA have higher needs than rural households (the value makes consumption and production comparable)
                    "tariff_avg":[1],    # determines the data for generating the tariffs, 1 2 or 3
                    "consumption_dev":[10.0],                    # determines the variability of consumptions
                    "bias_low":[0.0],"bias_high":[0.0], # bias for agents, low and high caste
                    "bias_mediator":[0.0]                 # bias for mediator, if 0 there is no bias, if None the bias is determined by the agents
                }}
test_N={             # bid splitting should increase efficiency
    "test_N":{**default_params,"bidsplit":False,"multibid":False,
                       'params':{**default_params['params']}}}
test_bids={             # bid splitting should increase efficiency
    "test_normal":{**default_params,"bidsplit":False,"multibid":False,
                       'params':{**default_params['params']}}, # no bias
    "test_bidsplit":{**default_params,"bidsplit":True,"multibid":False,
                     'params':{**default_params['params']}},
    "test_multibid":{**default_params,"bidsplit":False,"multibid":True,
                     'params':{**default_params['params']}}
    }
test_bias={          # bias should reduce efficiency and market access
    "test_bias_agents":{**default_params,"bidsplit":False,"multibid":False,
                       'params':{**default_params['params'],
                                 "N":Ns,"bias_low":[0.0],"bias_high":biases,"bias_mediator":[0.0]}}, # no mediator bias
    "test_bias_mediator":{**default_params,"bidsplit":False,"multibid":False,
                     'params':{**default_params['params'],
                               "N":Ns,"bias_low":[0.0],"bias_high":[0.0],"bias_mediator":biases}}} # no biased agents
test_bias_bidsplit={          # bias should reduce efficiency and market access
    "test_bias_split_agents":{**default_params,"bidsplit":True,"multibid":False,
                       'params':{**default_params['params'],
                                 "N":Ns,"bias_low":biases,"bias_high":biases,"bias_mediator":[0.0]}}, # no mediator bias
    "test_bias_split_mediator":{**default_params,"bidsplit":True,"multibid":False,
                     'params':{**default_params['params'],
                               "N":Ns,"bias_low":[0.0],"bias_high":[0.0],"bias_mediator":biases}}} # no biased agents
experiments={"exp_base":{**default_params,"bidsplit":False,"multibid":False,
                         'params':{**default_params['params'],
                                   "bias_low":[0.0],"bias_high":biases, # agents are biased
                                   "bias_mediator":[0.0] # Mediator is not biased
                                   }},
             "exp1":{**default_params,"bidsplit":False,"multibid":False,
                         'params':{**default_params['params'],
                                   "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                   "bias_mediator":[0.0] # Mediator is not biased
                                   }},
             "exp2":{**default_params,"bidsplit":True,"multibid":False, # bid splitting
                         'params':{**default_params['params'],
                                   "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                   "bias_mediator":[0.0] # Mediator is not biased
                                   }},
             "exp3":{**default_params,"bidsplit":False,"multibid":False, # no bid splitting
                         'params':{**default_params['params'],
                                   "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                   "bias_mediator":biases # Mediator is biased
                                   }},
             "exp4":{**default_params,"bidsplit":True,"multibid":False, # bid splitting
                         'params':{**default_params['params'],
                                   "bias_low":[0.0],"bias_high":[0.0], # agents are not biased
                                   "bias_mediator":biases # Mediator is biased
                                   }}}

for test,conf in test_bias.items():
    run_experiment(test,conf,"./output")
