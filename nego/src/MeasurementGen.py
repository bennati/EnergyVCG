import numpy as np
from src.MeasurementGen import BaseMeasurementGen
from src.utils import renormalize,positive_sampling,compute_stats
from nego.src.utilsnego import *
from numpy.random import choice
import pandas as pd
import os

# class NegoMeasurementGen():
#     def __init__(self):
#         np.random.seed()
#         pass

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         return [{"value":0,"timestep":timestep}]

# class MeasurementGenUniform(BaseMeasurementGen):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.n1=kwargs["n1"]
#         self.n2=kwargs["n2"]

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         ret=[{"value":np.random.uniform(self.n1,self.n2),"cost":0,"timestep":timestep,"agentID":i}
#              for i in range(len(population))]
#         return ret

# class MeasurementGenNormal(BaseMeasurementGen):
#     def __init__(self,*args, **kwargs):
#         super().__init__()
#         self.mu=kwargs["mu"]
#         self.s=3

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         ret=[{"production":np.random.normal(loc=self.mu,scale=self.s),
#               "consumption":np.random.normal(loc=self.mu,scale=self.s),
#               "timestep":timestep,"agentID":i,"tariff":np.random.uniform(low=0,high=5)}
#              for i in range(len(population))]
#         return ret

# class MeasurementGenBinomial(BaseMeasurementGen):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.mu1=kwargs["mu1"]
#         self.s1=1
#         self.mu2=kwargs["mu2"]
#         self.s2=1
#         # self.sep=kwargs["rich"]
#         self.produce_low = kwargs["buy_low"] # proportion of agents who can produce in lower caste
#         self.produce_high = kwargs["buy_high"] # proportion of agents who can produce in higher caste
#         self.caste=kwargs["low_caste"] # proportion of agents in low caste
#         self.biased_low=kwargs["bias_low"]  # proportion of biased agents among low caste
#         self.biased_high = kwargs["bias_high"] # proportion of biased agents among low caste
#         self.bias_mediator = kwargs["bias_degree"] # proportion of agents being biased by the mediator

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         ret=[{"production":(np.random.normal(loc=self.mu1,scale=self.s1)
#                        if i>len(population)*self.caste else
#                        np.random.normal(loc=self.mu2,scale=self.s2)),
#               "consumption":(np.random.normal(loc=self.mu1,scale=self.s1)
#                        if i>len(population)*self.caste else
#                        np.random.normal(loc=self.mu2,scale=self.s2)),
#               "tariff":np.random.uniform(1,5),"main_cost":0.1,
#               "social_type":(2 if i>len(population)*self.caste else 1),
#               "biased":(0 if i<len(population)*(1-self.caste)*(1-self.biased_high)
#                                 else(1 if i<len(population)*(1-self.caste)
#                                      else(0 if i<len(population)*((1-self.caste)+self.caste*(1-self.biased_low))
#                                           else 1))),
#               "bias_degree":(choice((True,False),1,p=(self.bias_mediator,(1-self.bias_mediator))))[0],
#               "cost":0,"timestep":timestep,"agentID":0,"type":None,
#               "old_production":0, "old_consumption":0}
#              for i in range(len(population))]
#         return ret

class MeasurementGenReal(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cons_dev=kwargs["consumption_dev"]
        self.tariffdev=0.5
        self.t=int(kwargs["T"])
        self.n=int(kwargs["N"])
        if self.t is None:
            print("setting t to default")
            self.t=1000
        self.caste=kwargs["low_caste"] # proportion of agents in low caste
        self.biased_low=kwargs["bias_low"]  # proportion of biased agents among low caste
        self.biased_high = kwargs["bias_high"] # proportion of biased agents among high caste
        self.bias_mediator = kwargs["bias_degree"] # probability of mediator being biased
        self.tariff_avg = kwargs["tariff_avg"]
        self.produce_avg = kwargs["produce_avg"]
        # self.chancer=kwargs["chance_rich"]
        # self.chancerp=kwargs["chance_poor"]
        datadir='datasets'
        self.tariff_data=pd.read_csv(os.path.join(datadir,"tariff.csv"))
        self.consumption_data=pd.read_csv(os.path.join(datadir,"Consumption_data.csv"))
        self.caste_byincome,byvillage=read_income_data(datadir)
        if self.caste is None:
            self.caste=byvillage['Dalit_prop'].mean()/100

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        if timestep>self.t:
            return None
        else:
            # production = [self.produce_avg*income[i]*8/24/20000 for i in range(len(population))] # TODO what are these constants?, why is there no randomness?
            ## TODO, why is the tariff generated from the third column while the documentation talks about the second column?
            ## compute current tariff
            tariff=self.tariff_data.ix[timestep%self.tariff_data.shape[0], # after 24 hours the day repeats
                                       "inrpriceperkwh"+str(int(self.tariff_avg))] # choose the column in the data
            ## compute castes, start from the data about proportion of Dalit individuals in rural villages
            castes=[np.random.uniform()<self.caste for _ in range(len(population))] # determine the caste, true means low caste
            incomes=compute_incomes(self.caste_byincome,castes) # compute incomes based on real data
            consumptions=compute_consumptions(self.consumption_data,self.cons_dev,incomes,self.caste_byincome.income_max.max())
            productions=[i*self.produce_avg for i in compute_productions(incomes)]
            ret=[{"consumption":consumptions[i],
                  "tariff":positive_sampling(float(tariff),self.tariffdev), # a value normally distributed around the value in the data
                  "social_type":(1 if caste else 2),
                  "production":productions[i],
                  "biased":is_biased(caste,self.biased_low,self.biased_high),
                  "bias_mediator":is_mediator_biased(self.bias_mediator),
                  # "chance_rich":np.random.uniform()<self.chancer, # TODO should being rich depend on the income?
                  "agentID":0, "income":incomes[i],
                  "main_cost":0.1,"cost":0,"timestep":timestep,"type":None,"threshold":-1}
                 for i,caste in enumerate(castes)]  # high class is 2, low class is 1, main_cost is maintenance cost
            return ret
