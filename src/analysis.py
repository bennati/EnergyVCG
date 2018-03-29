import pandas as pd
import functools
from utils import *

def subset_df(df,conditions=pd.Series()):
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
    res_decs=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("decisions")])
    # res_decs=pd.read_csv("./data/"+str(test)+"/decisions.csv.gz")
    res_decs["algorithm"]=l
    res_decs["cost_pop"]=[(0 if np.isnan(i) else i)  for i in res_decs["cost"]]
    decs_list.append(res_decs)
    res_percs=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("perception")])
    # res_percs=pd.read_csv("./data/"+str(test)+"/perceptions.csv.gz")
    res_percs["algorithm"]=l
    percs_list.append(res_percs)
    res_rews=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("reward")])
    # res_rews=pd.read_csv("./data/"+str(test)+"/rewards.csv.gz")
    res_rews["algorithm"]=l
    rews_list.append(res_rews)
    res_eval=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("evaluation")])
    # res_eval=pd.read_csv("./data/"+str(test)+"/evaluation.csv.gz")
    res_eval["algorithm"]=l
    eval_list.append(res_eval)
    varnames=[v for v in ["N","n1","n2"] if (v in res_eval.columns) and (len(res_eval[v].unique())>1)]
    varvalues=expandgrid({v:res_eval[v].unique() for v in varnames})
    try:
        losses=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("loss")])
        cols=losses.columns
        types=[np.unique(i) for i in zip(*[str(c).split("_") for c in cols])]
        lstyles=['-','--',':','-.']
        losses["bins"]=losses.index//100
        losses=compute_stats(losses,idx=["bins"])
        fig,ax=plt.subplots()
        # ax.set_yscale('log')
        cmap = plt.get_cmap('cubehelix_r')
        colors=[cmap(float(i+1)/(losses.shape[1]+1)) for i in range(len(types[0]))]
        # colorArtists = [plt.Line2D((0,1),(0,0), color=c) for c in colors]
        for d,c in zip(types[0],colors):
            names=([[str(d),"-"]]) if len(types)==1 else ([[str(d)+"_"+str(t),st] for t,st in zip(types[1],lstyles[:len(types[1])])])
            for name,st in names:
                ax.plot(losses.index,losses[name+"_mean"],color=c,linewidth=3,label=name,linestyle=st)
                ax.fill_between(losses.index,np.asarray(losses[name+"_mean"])-np.asarray(losses[name+"_ci"]),np.asarray(losses[name+"_mean"])+np.asarray(losses[name+"_ci"]),alpha=0.2,facecolor=c,linewidth=3,linestyle=st)
        ax.legend()
        fig.savefig("./plots/"+str(test)+"/loss.pdf",format='pdf')
        plt.close(fig)
        # plot_trend(losses,"bins","./plots/"+str(test)+"/loss.pdf",trends=cols)
    except Exception as e:
        print("cannot print losses "+str(e))
        losses=None
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
        f=functools.partial(subset_df,conditions=pd.Series({"timestep":int(res_decs["timestep"].max())}))
        stats_percs=compute_stats(f(res_percs),idx=[varname],columns=["value","cost"])
        plot_trend(stats_percs,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf")
        stats_decs=compute_stats(f(res_decs),idx=[varname],columns=["contribution","cost","cost_pop","contributed"])
        plot_trend(stats_decs,varname,"./plots/"+str(test)+"/decisions_"+str(varname)+".pdf")
        stats_eval=compute_stats(f(res_eval),idx=[varname],columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
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
    vars2plot=["prob"]
    # var2plot="yes"
    try:
        qtables=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("qtab")])
        # qtables=pd.read_csv("./data/"+str(test)+"/qtables.csv.gz")
        vars2plot=[v for v in vars2plot if v in qtables.columns]
        qtables_stats=compute_stats([qtables],idx=["state_val","state_cost"],columns=vars2plot)#+["num"])
        stats_q=compute_stats([qtables],idx=["state_val","state_cost"]+varnames,columns=vars2plot)#+["num"])
        stats_qa=compute_stats([qtables],idx=["state_val","state_cost","idx"]+varnames,columns=vars2plot)#+["num"])
        for var2plot in vars2plot:
            plot_trend(qtables_stats,"state_cost","./plots/"+str(test)+"/qtables_"+str(var2plot)+"_cost.pdf",yname="state_val",trends=[var2plot])
            plot_trend(qtables_stats,"state_val","./plots/"+str(test)+"/qtables_"+str(var2plot)+"_val.pdf",yname="state_cost",trends=[var2plot])
            for idx,p in varvalues.iterrows():
                pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                ## plot stats qtable
                try:
                    tmp=subset_df(stats_q,p)
                    f=lambda df,col: np.histogram2d(df["state_cost"],df["state_val"],weights=df[col+"_mean"],bins=[np.append(df["state_val"].unique(),[df["state_val"].max()+1]),np.append(df["state_cost"].unique(),[df["state_cost"].max()+1])])
                    heatmap_choice,xlabs,ylabs=f(tmp,var2plot)
                    plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_choice_"+str(var2plot)+"_"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                    ## plot individual qtables
                    tmp=subset_df(stats_qa,p)
                    prefix="./plots/"+str(test)+"/qtabs"
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)
                    for a in tmp["idx"].unique():
                        heatmap_choice,xlabs,ylabs=f(tmp[tmp["idx"]==a],var2plot)
                        plot_hmap(heatmap_choice,"Average qvalue associated to contribution","heat_q_choice_"+str(var2plot)+"_"+str(a)+"_"+pdesc+".pdf",prefix,xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
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
endtime=min([int(d["timestep"].max()) for d in decs_list])
rews_list=pd.concat(rews_list)
percs_list=pd.concat(percs_list)
decs_list=pd.concat(decs_list)
eval_list=pd.concat(eval_list)
for varname in varnames:
    stats_rews=compute_stats([rews_list],idx=[varname,"algorithm"],columns=["reward"])
    plot_trend(stats_rews,varname,"./plots/rewards_"+str(varname)+".pdf",yname="algorithm",trends=["reward"],xlab="Average value",ylab="Reward",font_size=16)
    # stats_percs=compute_stats([perc_list],idx=[varname,"algorithm"],columns=["value","cost"])
    # plot_trend(stats_percs,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf",yname="algorithm")
    f=functools.partial(subset_df,conditions=pd.Series({"timestep":endtime}))
    stats_decs=compute_stats([f(decs_list)],idx=[varname,"algorithm"],columns=["contribution","cost","cost_pop","contributed","privacy"])
    plot_trend(stats_decs,varname,"./plots/costs_volunteers_"+str(varname)+".pdf",yname="algorithm",trends=["cost"],xlab="Average value",ylab="Cost for volunteers",font_size=16)
    plot_trend(stats_decs,varname,"./plots/costs_global_"+str(varname)+".pdf",yname="algorithm",trends=["cost_pop"],xlab="Average value",ylab="Cost",font_size=16)
    plot_trend(stats_decs,varname,"./plots/priv_indiv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy"],xlab="Average value",ylab="Privacy loss",font_size=16)
    stats_eval=compute_stats([f(eval_list)],idx=[varname,"algorithm"],columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
    plot_trend(stats_eval,varname,"./plots/gini_"+str(varname)+".pdf",yname="algorithm",trends=["gini"],xlab="Average value",ylab="Inequality of contributions",font_size=16)
    plot_trend(stats_eval,varname,"./plots/success_"+str(varname)+".pdf",yname="algorithm",trends=["success"],xlab="Average value",ylab="Success rate",font_size=16)
    plot_trend(stats_eval,varname,"./plots/welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"],xlab="Average value",ylab="Social welfare",font_size=16)
    plot_trend(stats_eval,varname,"./plots/eff_"+str(varname)+".pdf",yname="algorithm",trends=["efficiency"],xlab="Average value",ylab="Efficiency",font_size=16)
    # plot_trend(stats_eval,varname,"./plots/priv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy"])
