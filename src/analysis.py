import pandas as pd
import functools
from utils import *

def subset_df(df,conditions):
    if conditions.empty:
        ret=df
    else:
        ret=df[functools.reduce(np.logical_and,[(df[k]==v) for k,v in zip(conditions.index,conditions)])] # select only a subset of the table matching the parameters
        ret.reset_index()
    return ret

tests=[["qlearn_nrel","Decentr."],["mandatory_nrel","Baseline"],["knapsack_nrel","Centr."]]
rews_list=[]
percs_list=[]
decs_list=[]
eval_list=[]
### generate individual plots
for test,l in tests:
    print(test)
    if not os.path.exists("plots/"+str(test)):
        os.makedirs("plots/"+str(test))
    res_decs=pd.read_csv("./data/"+str(test)+"/decisions.csv.gz")
    res_decs["algorithm"]=l
    res_decs["cost_pop"]=[(0 if np.isnan(i) else i)  for i in res_decs["cost"]]
    decs_list.append(res_decs)
    res_percs=pd.read_csv("./data/"+str(test)+"/perceptions.csv.gz")
    res_percs["algorithm"]=l
    percs_list.append(res_percs)
    res_rews=pd.read_csv("./data/"+str(test)+"/rewards.csv.gz")
    res_rews["algorithm"]=l
    rews_list.append(res_rews)
    res_eval=pd.read_csv("./data/"+str(test)+"/evaluation.csv.gz")
    res_eval["algorithm"]=l
    eval_list.append(res_eval)
    varnames=[v for v in ["N","n1","n2"] if (v in res_eval.columns) and (len(res_eval[v].unique())>1)]
    varvalues=expandgrid({v:res_eval[v].unique() for v in varnames})
    try:
    contrib_hist=pd.read_csv("./data/"+str(test)+"/contrib_hist.csv.gz")
    except:
        contrib_hist=None
    # varnames=[c for c in contrib_hist.columns if c not in ["value","cnt"]]
    # varvalues=expandgrid({v:contrib_hist[v].unique() for v in varnames})
    try:
    stats_gini_contribs=pd.read_csv("./data/"+str(test)+"/stats_gini_contribs.csv.gz")
    except:
        stats_gini_contribs=None
    stats_evalt=compute_stats(res_eval,idx=["timestep"],columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
    plot_measures(stats_evalt,"timestep","./plots/"+str(test)+"/eval_"+str("time")+".pdf")
    ### now move to computing statistics that aggregate on one of the parameters ###
    for varname in varnames:
        if stats_gini_contribs is not None:
        stats_gini=compute_stats([stats_gini_contribs],[varname],columns=["Contributors","Values"]) # average across repetitions
        plot_trend(stats_gini,varname,"./plots/"+str(test)+"/gini_"+str(varname)+".pdf")
        stats_rews=compute_stats(res_rews,idx=[varname],columns=["reward"])
        plot_trend(stats_rews,varname,"./plots/"+str(test)+"/rewards_"+str(varname)+".pdf")
        stats_percs=compute_stats(res_percs,idx=[varname],columns=["value","cost"])
        plot_trend(stats_percs,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf")
        stats_decs=compute_stats(res_decs,idx=[varname],columns=["contribution","cost","cost_pop","contributed"])
        plot_trend(stats_decs,varname,"./plots/"+str(test)+"/decisions_"+str(varname)+".pdf")
        stats_eval=compute_stats(res_eval,idx=[varname],columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
        plot_measures(stats_eval,varname,"./plots/"+str(test)+"/eval_"+str(varname)+".pdf")
        if contrib_hist is not None:
        stats_contrib_hist=compute_stats(contrib_hist,idx=[varname,"value"],columns=["cnt"])
        plot_trend(stats_contrib_hist,"value","./plots/"+str(test)+"/contrib_hist_"+str(varname)+".pdf",yname=varname)

    ### now compute statistics for each parameter configuration, aggregating only on repetitions ###
    stats_t=compute_stats(res_eval,idx=["timestep"]+varnames,columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
    for idx,p in varvalues.iterrows():
        pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        # temporal evolution of measures
        tmp=subset_df(stats_t,p)
        plot_measures(tmp,"timestep","./plots/"+str(test)+"/time_"+pdesc+".pdf")
        # distribution of contributions
        # tmp=subset_df(stats_contrib_hist2,p)
        # plot_trend(tmp,"value","./plots/"+str(test)+"/contrib_hist_"+pdesc+".pdf")

    ## compute qtable heatmaps
    var2plot="prob"
    # var2plot="yes"
    try:
        qtables=pd.read_csv("./data/"+str(test)+"/qtables.csv.gz")
        qtables_stats=compute_stats([qtables],idx=["state_val","state_cost"],columns=[var2plot,"num"])
        plot_trend(qtables_stats,"state_cost","./plots/"+str(test)+"/qtables_cost.pdf",yname="state_val",trends=[var2plot])
        plot_trend(qtables_stats,"state_val","./plots/"+str(test)+"/qtables_val.pdf",yname="state_cost",trends=[var2plot])
        stats_q=compute_stats([qtables],idx=["state_val","state_cost"]+varnames,columns=[var2plot,"num"])
        stats_qa=compute_stats([qtables],idx=["state_val","state_cost","idx"]+varnames,columns=[var2plot,"num"])
        for idx,p in varvalues.iterrows():
            pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
            ## plot stats qtable
            try:
            tmp=subset_df(stats_q,p)
            f=lambda df,col: np.histogram2d(df["state_cost"],df["state_val"],weights=df[col+"_mean"],bins=[np.append(df["state_val"].unique(),[df["state_val"].max()+1]),np.append(df["state_cost"].unique(),[df["state_cost"].max()+1])])
                heatmap_choice,xlabs,ylabs=f(tmp,var2plot)
            plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_choice"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
            ## plot individual qtables
            tmp=subset_df(stats_qa,p)
            prefix="./plots/"+str(test)+"/qtabs"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            for a in tmp["idx"].unique():
                    heatmap_choice,xlabs,ylabs=f(tmp[tmp["idx"]==a],var2plot)
                plot_hmap(heatmap_choice,"Average qvalue associated to contribution","heat_q_choice_"+str(a)+"_"+pdesc+".pdf",prefix,xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
            except:
                print("unable to plot heatmap for p: "+str(p))
            # heatmap_count,xlabs,ylabs=f(tmp,"num")
            # plot_hmap(heatmap_count,"Average number of occurrences of a state",str(test)+"/heat_q_count"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
            ## compute histograms
            q_exp=subset_df(qtables,p) # subset with current experimental conditions
            plot_qtable_hist(q_exp,"./plots/"+str(test)+"/qhist_"+pdesc+".pdf","state_val","state_cost",var2plot,str(dict(p)))
            prefix="./plots/"+str(test)+"/heatmaps"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            plot_qtable_heat(q_exp,prefix+"/qheat_"+pdesc,"state_val","state_cost",var2plot)
    except:
        print("qtables not found")

### generate comparison plots
rews_list=pd.concat(rews_list)
percs_list=pd.concat(percs_list)
decs_list=pd.concat(decs_list)
eval_list=pd.concat(eval_list)
for varname in varnames:
    stats_rews=compute_stats([rews_list],idx=[varname,"algorithm"],columns=["reward"])
    plot_trend(stats_rews,varname,"./plots/rewards_"+str(varname)+".pdf",yname="algorithm",trends=["reward"])
    # stats_percs=compute_stats([perc_list],idx=[varname,"algorithm"],columns=["value","cost"])
    # plot_trend(stats_percs,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf",yname="algorithm")
    stats_decs=compute_stats([decs_list],idx=[varname,"algorithm"],columns=["contribution","cost","cost_pop","contributed","privacy"])
    plot_trend(stats_decs,varname,"./plots/costs_volunteers_"+str(varname)+".pdf",yname="algorithm",trends=["cost"])
    plot_trend(stats_decs,varname,"./plots/costs_global_"+str(varname)+".pdf",yname="algorithm",trends=["cost_pop"])
    plot_trend(stats_decs,varname,"./plots/priv_indiv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy"])
    stats_eval=compute_stats([eval_list],idx=[varname,"algorithm"],columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
    plot_trend(stats_eval,varname,"./plots/gini_"+str(varname)+".pdf",yname="algorithm",trends=["gini"])
    plot_trend(stats_eval,varname,"./plots/success_"+str(varname)+".pdf",yname="algorithm",trends=["success"])
    plot_trend(stats_eval,varname,"./plots/welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"])
    plot_trend(stats_eval,varname,"./plots/eff_"+str(varname)+".pdf",yname="algorithm",trends=["efficiency"])