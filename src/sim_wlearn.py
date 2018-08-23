from DecisionLogic import BaseDecisionLogic
from RewardLogic import BaseRewardLogic
from utils import *
from brains import *
import itertools
import numpy as np

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

class RewardLogicW(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=10
        self.damage=-10
        self.last_gini=0

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in decisions])
        contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
        outcome=success(thresh,contribs)
        self.last_gini=gini([d["contributed"] for d in decisions])
        ratios={d["agentID"]:{"contributed":d["avg_hist_contributed"]-d["hist_contributed"],"contribution":d["avg_hist_contribution"]-d["hist_contribution"]} for d in decisions}
        # contr1=[v["contributed"] for k,v in ratios.items()]
        # contr2=[v["contribution"] for k,v in ratios.items()]
        # ratios={k:{"contributed":renormalize(v["contributed"],[min(contr1),max(contr1)],[0,1]),
        #            "contribution":renormalize(v["contribution"],[min(contr2),max(contr2)],[0,1])} for k,v in ratios.items()} # contain values or qvalues will explode
        # print("gini for contributions "+str([d["contributed"] for d in decisions])+" is "+str(self.last_gini))
        # self.last_gini=int(renormalize(self.last_gini,[0,1],[0,gini_bin_no])) # binarize
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
        if outcome==1:
            ret=-costs+self.benefit
        else:
            # print("unsuccessful")
            ret=-costs+self.damage
        # ret=costs
        ret=[renormalize(r,[self.damage-self.model.measurement_fct.n2,self.benefit],[-1,1]) for r in ret] # contain values or qvalues will explode
        ret=[{"agentID": d["agentID"],"reward":r,"gini":self.last_gini,"contributed":ratios[d["agentID"]]["contributed"],"contribution":ratios[d["agentID"]]["contribution"]} for r,d in zip(ret,decisions)]
        return ret

class RewardLogicWavg(BaseRewardLogic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit=10
        self.damage=-10

    def get_rewards(self,decisions):
        """
        The threshold is randomly generated around the average contribution
        """
        thresh=max([p["threshold"] for p in decisions])
        contribs=np.sum([d["contribution"] for d in decisions if d["contributed"]])
        outcome=success(thresh,contribs)
        # ratios={d["agentID"]:{"contributed":d["avg_hist_contributed"]-d["hist_contributed"],"contribution":d["avg_hist_contribution"]-d["hist_contribution"]} for d in decisions}
        # contr1=[v["contributed"] for k,v in ratios.items()]
        # contr2=[v["contribution"] for k,v in ratios.items()]
        # ratios={k:{"contributed":renormalize(v["contributed"],[min(contr1),max(contr1)],[0,1]),
        #            "contribution":renormalize(v["contribution"],[min(contr2),max(contr2)],[0,1])} for k,v in ratios.items()} # contain values or qvalues will explode
        # print("gini for contributions "+str([d["contributed"] for d in decisions])+" is "+str(self.last_gini))
        # self.last_gini=int(renormalize(self.last_gini,[0,1],[0,gini_bin_no])) # binarize
        costs=np.array([(d["cost"] if d["contributed"] else 0) for d in decisions])
        if outcome==1:
            ret=-costs+self.benefit
        else:
            # print("unsuccessful")
            ret=-costs+self.damage
        # ret=costs
        ret=[renormalize(r,[self.damage-self.model.measurement_fct.n2,self.benefit],[-1,1]) for r in ret] # contain values or qvalues will explode
        ret=[r+d["avg_hist_contributed"]-d["hist_contributed"] for r,d in zip(ret,decisions)]
        print(np.asarray(ret)-(np.asarray(ret)+np.asarray(d["avg_hist_contributed"]-d["hist_contributed"])))
        ret=[{"agentID": d["agentID"],"reward":r} for r,d in zip(ret,decisions)]
        return ret

class DecisionLogicSupervisorW(BaseDecisionLogic):
    """
    Returns a constant decision
    """

    def __init__(self,model,training=True,**kwargs):
        super().__init__(model)
        self.public_info={i:{"hist_contributed":0,"hist_contribution":0,"avg_hist_contributed":0,"avg_hist_contribution":0} for i in range(self.model.N)}
        self.time=1

    def get_decision(self,perceptions):
        decs=[a.get_decision() for a in self.model.schedule.agents]
        idxs=[a.unique_id for a in self.model.schedule.agents]
        tmp1=pd.DataFrame(data={"contributed":decs,"agentID":idxs})
        tmp=pd.merge(pd.DataFrame(perceptions),tmp1,on=["agentID"])
        self.time=tmp["timestep"].max()+1
        ## update public info
        for idx,d in tmp.iterrows():
            if d["contributed"]:
                now=self.public_info[d["agentID"]]
                self.public_info[d["agentID"]].update({"hist_contributed":now["hist_contributed"]+1})
                self.public_info[d["agentID"]].update({"hist_contribution":now["hist_contribution"]+d["value"]})
        ## assign public info
        for k,v in self.public_info.items():
            v.update({"avg_hist_contributed":np.mean([i["hist_contributed"] for j,i in self.public_info.items() if j!=k])})
            v.update({"avg_hist_contribution":np.mean([i["hist_contribution"] for j,i in self.public_info.items() if j!=k])})
        # for d in decisions:
        #     d.update({"own_hist_contributed":self.public_info[d["agentID"]]["contributed"],
        #               "own_hist_contribution":self.public_info[d["agentID"]]["value"],
        #               "avg_hist_contributed":np.mean([v["contributed"] for k,v in self.public_info.items() if k!=d["agentID"]]),
        #               "avg_hist_contribution":np.mean([v["value"] for k,v in self.public_info.items() if k!=d["agentID"]]),
        #     })
        tmp=pd.merge(tmp,pd.DataFrame(self.public_info).T,left_on="agentID",right_index=True)
        # means without the element
        #print(decs)
        self.act=[{**self.get_public_info()[r[1]["agentID"]],"contribution":(r[1]["value"] if r[1]["contributed"] else np.nan),"cost":(r[1]["cost"] if r[1]["contributed"] else np.nan),"privacy":(1 if r[1]["contributed"] else 0),"agentID":r[1]["agentID"],"contributed":r[1]["contributed"],"timestep":r[1]["timestep"],"threshold":r[1]["threshold"]} for r in tmp.iterrows()]
        del tmp,decs,idxs,tmp1
        return self.act

    def get_qtable(self):
        qtabs,losses=zip(*[a.decision_fct.get_qtable() for a in self.model.schedule.agents])
        try:
            losses={k:v for b in losses for k,v in b.items()} # flatten
            losses=pd.DataFrame(data=losses)
        except:
            losses=None
        return pd.concat(qtabs),losses

    def get_public_info(self):
        return {k:{"avg_hist_contribution":v["avg_hist_contribution"]/self.time,
                   "avg_hist_contributed":v["avg_hist_contributed"]/self.time,
                   "hist_contribution":v["hist_contribution"]/self.time,
                   "hist_contributed":v["hist_contributed"]/self.time} for k,v in self.public_info.items()}

class DecisionLogicWlearn(BaseDecisionLogic):
    def __init__(self,model,gamma=0.0,alpha=0.001,training=True):
        super().__init__(model)
        self.bins=[np.arange(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2),
                   np.arange(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2),
                   np.arange(0,1,0.3),
                   np.arange(0,1,0.3)]
        self.states=list(itertools.product(*self.bins)) # all possible states
        self.actions=[0,1]
        self.act=1
        self.qlearner=DQlearner(self.states,self.actions,gamma=0.0,alpha=0.001,n_features=4,learn_step=10,batch_size=10)
        if training:
            self.train_rew()   # pretraining
        self.wlearner=Wlearner(self.states,gamma=gamma)
        # self.qlearner_gini=DQlearner([self.model.model.measurement_fct.n1,self.model.model.measurement_fct.n2],self.actions,gamma=gamma,n_features=len(self.states[0]))
        # self.wlearner_gini=Wlearner(self.states,gamma=gamma)
        self.qlearner_hist=DQlearner(self.states,self.actions,gamma=0.2,alpha=0.001,n_features=4,learn_step=10,batch_size=10)
        if training:
            self.train_hist()   # pretraining
        self.wlearner_hist=Wlearner(self.states,gamma=gamma)
        self.last_info=self.model.model.decision_fct.get_public_info()[self.model.unique_id]

    def get_decision(self,perceptions):
        # print("--------------------")
        # self.last_gini=self.model.model.reward_fct.last_gini
        self.last_info=self.model.model.decision_fct.get_public_info()[self.model.unique_id]
        current=self.get_current_state()+tuple([self.last_info["hist_contributed"],self.last_info["avg_hist_contributed"]])
        binned_s=self.binarize_state(current,self.bins)
        # print("evaluating state "+str(current))
        policies=[[q.get_decision(current),w.get_decision(binned_s)] for q,w in [[self.qlearner,self.wlearner],[self.qlearner_hist,self.wlearner_hist]]]
        # print(policies)
        # choices,ws=[i.get_decision(current) for i in [self.qlearner,self.wlearner]]
        # choices_gini,ws_gini=[i.get_decision(self.last_gini) for i in [self.qlearner_gini,self.wlearner_gini]]
        # actions=[choices,choices_gini]
        # print("selected actions "+str(actions))
        self.winning_policy=np.argmax([w for action,w in policies])
        # self.winning_policy=0 #np.argmax([ws,ws_gini]) #TODO
        self.act=policies[self.winning_policy][0]
        # print("Ws are "+str([policies[0][1],policies[1][1]])+": winning policy is "+str(self.winning_policy))
        # self.act=self.qlearner.get_decision(current)
        assert(self.act in self.actions)
        return self.act
        # self.act=self.qlearner_hist.get_decision(current)
        # return self.act

    def feedback(self,perceptions,reward):
        assert(reward["agentID"]==self.model.unique_id)
        s=self.get_current_state()
        self.reward=reward["reward"]
        # next_gini=reward["gini"]
        # gini_reward=1-next_gini
        info=self.model.model.decision_fct.get_public_info()[self.model.unique_id]
        hist_reward=info["avg_hist_contributed"]-info["hist_contributed"]
        # print("Giving reward "+str(self.reward)+" and gini "+str(gini_reward)+" for state "+str(current))
        current=s+tuple([self.last_info["hist_contributed"],self.last_info["avg_hist_contributed"]])
        nxt=s+tuple([info["hist_contributed"],info["avg_hist_contributed"]])
        self.qlearner.learn(current,nxt,self.act,reward["reward"])
        self.qlearner_hist.learn(current,nxt,self.act,hist_reward)
        # update wlearning, only policy that lost
        binned_s=self.binarize_state(current,self.bins)
        # print("updating policy "+str(self.winning_policy)+" with rewards "+str(self.reward)+" and "+str(hist_reward))
        if self.winning_policy: # gini won
            self.wlearner.learn(binned_s,nxt,self.reward,self.qlearner.get_qvalues(current).max(),0)
        else:
            self.wlearner_hist.learn(binned_s,nxt,hist_reward,self.qlearner_hist.get_qvalues(current).max(),0)

    def get_qtable(self):
        asd,l=self.qlearner.get_qtable()
        asd.rename(columns={0:"no_rew",1:"yes_rew"},inplace=True)
        loss={str(self.model.unique_id)+"_rew":l}
        asd1,_=self.wlearner.get_qtable()
        asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        asd1,l=self.qlearner_hist.get_qtable()
        asd1.rename(columns={0:"no_hist",1:"yes_hist"},inplace=True)
        loss.update({str(self.model.unique_id)+"_hist":l})
        asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        asd1,_=self.wlearner_hist.get_qtable()
        asd1.rename(columns={"w":"whist"},inplace=True)
        asd=pd.merge(asd,asd1,left_index=True,right_index=True)
        asd["yes"]=asd["yes_rew"]
        asd.loc[asd["whist"]>asd["w"],"yes"]=asd["yes_hist"]
        asd["no"]=asd["no_rew"]
        asd.loc[asd["whist"]>asd["w"],"no"]=asd["no_hist"]
        asd["idx"]=self.model.unique_id
        print(asd)
        return asd,loss
        # return self.qlearner_hist.get_qtable()

    # def get_qcount(self):
    #     return self.qlearner.get_qcount()

    def get_current_state(self):
        return (self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])

    def binarize_state(self,s,bins):
        assert(len(s)==len(bins))
        idxs=[np.argmin(abs(b-i)) for i,b in zip(s,bins)]
        ret=tuple([bins[i][v] for i,v in enumerate(idxs)])
        assert(ret in self.states)
        return ret

    def train_rew(self):
        """
        Pretrain the network to contribute
        """
        for s in self.states:
            for _ in range(100):
                self.qlearner.learn(s,[0]*len(s),0,0)
                self.qlearner.learn(s,[0]*len(s),1,1) # contributing is better
        self.qlearner.cost_his = []

    def train_hist(self):
        """
        Pretrain the network to contribute
        """
        for s in self.states:
            own_hist=s[2]
            other_hist=s[3]
            rew=1-own_hist-other_hist*0.7
            for _ in range(100):
                self.qlearner_hist.learn(s,[0]*len(s),0,0)
                self.qlearner_hist.learn(s,[0]*len(s),1,rew) # contributing is better
        self.qlearner_hist.cost_his = []
