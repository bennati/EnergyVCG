import numpy as np
from src.MeasurementGen import BaseMeasurementGen
from src.utils import renormalize
from src.utils import positive_sampling
import csv
from numpy.random import choice
import pandas as pd
import os

class NegoMeasurementGen():
    def __init__(self):
        np.random.seed()
        pass

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        return [{"value":0,"timestep":timestep}]

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
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=0.5
        self.t=int(kwargs["T"])
        if self.t is None:
            print("setting t to default")
            self.t=1000
        self.produce_low = kwargs["buy_low"] # proportion of agents who can produce in lower caste
        self.produce_high = kwargs["buy_high"] # proportion of agents who can produce in higher caste
        self.caste=kwargs["low_caste"] # proportion of agents in low caste
        self.biased_low=kwargs["bias_low"]  # proportion of biased agents among low caste
        self.biased_high = kwargs["bias_high"] # proportion of biased agents among low caste
        self.bias_mediator = kwargs["bias_degree"] # proportion of agents being biased by the mediator
        self.tariff_avg = kwargs["tariff_avg"]
        self.produce_avg = kwargs["produce_avg"]
        self.min_income=kwargs["min_income"]
        self.max_income=kwargs["max_income"]
        # self.chancer=kwargs["chance_rich"]
        # self.chancerp=kwargs["chance_poor"]
        datadir='data'
        self.tariff_data=pd.read_csv(os.path.join(datadir,"tariff.csv"),names=["timestamp","tariff_rate","inrpriceperkwh"+str(int(self.tariff_avg))]).reset_index(drop=True)


    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        if timestep>self.t:
            return None
        else:
            mi=self.min_income
            ma=self.max_income
            assert(mi>=0)
            assert(ma>mi)
            # income_array = renormalize(np.arange(mi,ma,(ma-mi)/len(population)),[mi,ma],[0,1])
            income_array = np.arange(0,1,1/len(population)) # TODO uniformly spaced vector from 0 to 1? is it not supposed to be a uniform distribution?
            income = [np.random.uniform(mi,ma) for i in range(len(population))] # a uniformly distributed income for each agent
            production = [self.produce_avg*income[i]*8/24/20000 for i in range(len(population))] # TODO what are these constants?, why is there no randomness?
            # with open('data/tariff.csv') as csvfile:
            #     has_header = csv.Sniffer().sniff(csvfile.readline())
            #     csvfile.seek(0)
            #     readCSV = csv.DictReader(csvfile,fieldnames=["timestamp","tariff_rate","inrpriceperkwh"+str(int(self.tariff_avg))])
            #     if has_header:
            #         next(readCSV)
            #     data = [row for row in readCSV]
            #     tariff = data[timestep]["inrpriceperkwh"+str(int(self.tariff_avg))]
            ## TODO, why is the tariff generated from the third column while the documentation talks about the second column?
            tariff=self.tariff_data.ix[timestep,"inrpriceperkwh"+str(int(self.tariff_avg))]
            is_low_caste= lambda i: i<len(population)*self.caste
            is_high_caste= lambda i: i>len(population)*self.caste
            is_productive = lambda i: (i<len(population)*self.caste*self.produce_low # proportion of productive low-caste individuals
                                  if is_low_caste(i) else
                                  i<len(population)*(self.caste)+ # i is the index in the population, so we must start after all low-caste individuals
                                  len(population)*(1-self.caste)*self.produce_high) # proportion of productive high-caste individuals
            # TODO should the bias be assigned probabilistically?
            is_biased = lambda i: (i<len(population)*self.caste*self.biased_low # proportion of biased low-caste individuals
                              if is_low_caste(i) else
                              i<len(population)*(self.caste)+ # i is the index in the population, so we must start after all low-caste individuals
                              len(population)*(1-self.caste)*self.biased_high) # proportion of biased high-caste individuals
            ## debug
            # v=np.random.uniform(len(population),size=1000)
            # p=np.array(list(map(is_productive,v)))
            # l=np.array(list(map(is_low_caste,v)))
            # h=np.array(list(map(is_high_caste,v)))
            # pl=sum(p & l)/1000 # equals to self.caste*produce_low
            # ph=sum(p & h)/1000 # equals to (1-self.caste)*self.produce_high
            bias_mediator=(np.random.uniform()<self.bias_mediator if self.bias_mediator!=0 else None) # the mediator is not biased if the bias is 0
            ret=[{"consumption":positive_sampling(
                (self.mu2 if is_high_caste(i) else self.mu1)
                ,self.s1),
                  "tariff":positive_sampling(float(tariff),self.s2), # a value normally distributed around the value in the data
                  "social_type":(2 if is_high_caste(i) else 1),
                  "production":production[i] if is_productive(i) else 0,
                  "biased":1 if is_biased(i) else 0,
                  "bias_degree":bias_mediator,
                  # "chance_rich":np.random.uniform()<self.chancer, # TODO should being rich depend on the income?
                  # "chance_average":np.random.uniform()<self.chancerp,
                  "agentID":0, "income":renormalize(income[i],[mi,ma],[0,1])[0],
                  "income_excess":(renormalize(income[i],[mi,ma],[0,1])[0]-income_array[i]),
                  "main_cost":0.1,"cost":0,"timestep":timestep,"type":None}
                 for i in range(len(population))]  # high class is 2, low class is 1, main_cost is maintenance cost
            return ret
