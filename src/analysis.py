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

def plot_algs(df,xname,filename,trends=None,yname=None,lstyles=['-','--',':','-.'],colors=None,font_size=12,ylab=None,xlab=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    ax.set_xlabel(xlab or xname,fontsize=font_size)
    ax.set_ylabel(ylab or yname,fontsize=font_size)
    if yname is None:
        data=[(df,None)]
    else:
        data=[(df[df[yname]==i],i) for i in df[yname].unique()]
    if colors is None:
        cmap = plt.get_cmap('cubehelix_r')
        colors=[cmap(float(i+1)/(len(data)+1)) for i in range(len(data))]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 1.1])
    colorArtists = []
    for (d,l),c in zip(data,colors):
        x=d[xname]
        sty=d["lsty"].unique()[0]
        colorArtists.append(plt.Line2D((0,1),(0,0), color=c, marker='', linestyle=sty,linewidth=3))
        for y in trends:
            lab=(y if l is None else y+"; "+yname+"="+str(l))
            ax.plot(x,d[y+"_mean"],label=lab,linestyle=sty,color=c,linewidth=3)
            ax.fill_between(x,np.asarray(d[y+"_mean"])-np.asarray(d[y+"_ci"]),np.asarray(d[y+"_mean"])+np.asarray(d[y+"_ci"]),alpha=0.2,linestyle=sty,facecolor=c,linewidth=3)
    ax.add_artist(plt.legend(colorArtists,[l for d,l in data],loc='upper center', bbox_to_anchor=(0.5, 1.18),fancybox=True, shadow=True, ncol=len(data),fontsize=font_size))
    # plt.legend()
    plt.setp(ax.xaxis.get_majorticklabels(),fontsize=font_size)
    plt.setp(ax.yaxis.get_majorticklabels(),fontsize=font_size)
    fig.savefig(filename,format='pdf')
    plt.close(fig)
fprefix="nrel"
tests=[["nrel_mand","Full","--"],["nrel_def","No","--"],["nrel_knap","Cen.","-"],["nrel_Q","Dec.","-"]]
res_decs=None
res_percs=None
res_eval=None
res_rews=None
rews_list=[]
percs_list=[]
decs_list=[]
eval_list=[]
gini_contribs=[]
### generate individual plots
for test,l,sty in tests:
    print(test)
    if not os.path.exists("plots/"+str(test)):
        os.makedirs("plots/"+str(test))
    res_decs=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("decisions")])
    # res_decs=pd.read_csv("./data/"+str(test)+"/decisions.csv.gz")
    res_decs["algorithm"]=l
    res_decs["lsty"]=sty
    res_decs["cost_pop"]=[(0 if np.isnan(i) else i)  for i in res_decs["cost"]]
    decs_list.append(res_decs)
    res_percs=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("perception")])
    # res_percs=pd.read_csv("./data/"+str(test)+"/perceptions.csv.gz")
    res_percs["algorithm"]=l
    res_decs["lsty"]=sty
    percs_list.append(res_percs)
    res_rews=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("reward")])
    # res_rews=pd.read_csv("./data/"+str(test)+"/rewards.csv.gz")
    res_rews["algorithm"]=l
    res_decs["lsty"]=sty
    rews_list.append(res_rews)
    res_eval=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("evaluation")])
    # res_eval=pd.read_csv("./data/"+str(test)+"/evaluation.csv.gz")
    res_eval["algorithm"]=l
    res_eval["lsty"]=sty
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
        stats_gini_contribs["algorithm"]=l
        stats_gini_contribs["lsty"]=sty
        gini_contribs.append(stats_gini_contribs)
    except:
        stats_gini_contribs=None
    print("done reading files")
    stats_evalt=compute_stats(res_eval,idx=["timestep"],columns=["gini","gini_cost","cost_pop","efficiency","social_welfare","success","num_contrib"])
    plot_measures(stats_evalt,"timestep","./plots/"+str(test)+"/eval_"+str("time")+".pdf")
    ### now move to computing statistics that aggregate on one of the parameters ###
    for varname in varnames:
        print("plotting stats for var "+str(varname))
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
    print("plotting individual params")
    stats_t=compute_stats(res_eval,idx=["timestep"]+varnames,columns=["gini","cost_pop","efficiency","social_welfare","success","num_contrib"])
    for idx,p in varvalues.iterrows():
          print("plotting params "+str(dict(p)))
        pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        # temporal evolution of measures
        tmp=subset_df(stats_t,p)
        plot_measures(tmp,"timestep","./plots/"+str(test)+"/time_"+pdesc+".pdf")
        # distribution of contributions
        # tmp=subset_df(stats_contrib_hist2,p)
        # plot_trend(tmp,"value","./plots/"+str(test)+"/contrib_hist_"+pdesc+".pdf")

### generate comparison plots
print("generating comparison plots at times "+str([int(d["timestep"].max()) for d in decs_list]))
rews_list=pd.concat(rews_list)
percs_list=pd.concat(percs_list)
decs_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in decs_list])
eval_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in eval_list])
gini_contribs=pd.concat(gini_contribs)
for varname in varnames:
    print("plotting var "+str(varname))
    stats_rews=compute_stats([rews_list],idx=[varname,"algorithm","lsty"],columns=["reward"])
    plot_algs(stats_rews,varname,"./plots/wtest_rewards_"+str(varname)+".pdf",yname="algorithm",trends=["reward"],xlab="Average value",ylab="Reward",font_size=16)
    # stats_percs=compute_stats([perc_list],idx=[varname,"algorithm","lsty"],columns=["value","cost"])
    # plot_algs(stats_percs,varname,"./plots/"+str(test)+"/perceptions_"+str(varname)+".pdf",yname="algorithm")
    stats_decs=compute_stats([decs_list],idx=[varname,"algorithm","lsty"],columns=["contribution","cost","cost_pop","contributed","privacy"])
    stats_decs["privacy_inv_mean"]=1-stats_decs["privacy_mean"]
    stats_decs["privacy_inv_ci"]=stats_decs["privacy_ci"]
    plot_algs(stats_decs,varname,"./plots/"+fprefix+"_costs_volunteers_"+str(varname)+".pdf",yname="algorithm",trends=["cost"],xlab="Average value",ylab="Cost for volunteers",font_size=16)
    plot_algs(stats_decs,varname,"./plots/"+fprefix+"_costs_global_"+str(varname)+".pdf",yname="algorithm",trends=["cost_pop"],xlab="Average value",ylab="Cost",font_size=16)
    plot_algs(stats_decs,varname,"./plots/"+fprefix+"_priv_inv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy_inv"],xlab="Average value",ylab="Privacy",font_size=16)
    stats_eval=compute_stats([eval_list],idx=[varname,"algorithm","lsty"],columns=["gini","gini_cost","cost_pop","efficiency","social_welfare","success","num_contrib"])
    stats_eval["gini_inv_mean"]=1-stats_eval["gini_mean"]
    stats_eval["gini_inv_ci"]=stats_eval["gini_ci"]
    plot_algs(stats_eval,varname,"./plots/"+fprefix+"_giniinv_"+str(varname)+".pdf",yname="algorithm",trends=["gini_inv"],xlab="Average value",ylab="Equality of contributions",font_size=16)
    plot_algs(stats_eval,varname,"./plots/"+fprefix+"_ginicost_"+str(varname)+".pdf",yname="algorithm",trends=["gini_cost"],xlab="Average value",ylab="Inequality of contributions",font_size=16)
    plot_algs(stats_eval,varname,"./plots/"+fprefix+"_success_"+str(varname)+".pdf",yname="algorithm",trends=["success"],xlab="Average value",ylab="Success rate",font_size=16)
    plot_algs(stats_eval,varname,"./plots/"+fprefix+"_welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"],xlab="Average value",ylab="Social welfare",font_size=16)
    plot_algs(stats_eval,varname,"./plots/"+fprefix+"_eff_"+str(varname)+".pdf",yname="algorithm",trends=["efficiency"],xlab="Average value",ylab="Efficiency",font_size=16)
    # plot_algs(stats_eval,varname,"./plots/priv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy"])
    stats_gini=compute_stats([gini_contribs],[varname,"algorithm","lsty"],columns=["Contributors","Values","cost"]) # average across repetitions
    stats_gini["contr_inv_mean"]=1-stats_gini["Contributors_mean"]
    stats_gini["contr_inv_ci"]=stats_gini["Contributors_ci"]
    stats_gini["val_inv_mean"]=1-stats_gini["Values_mean"]
    stats_gini["val_inv_ci"]=stats_gini["Values_ci"]
    stats_gini["cost_inv_mean"]=1-stats_gini["cost_mean"]
    stats_gini["cost_inv_ci"]=stats_gini["cost_ci"]
    plot_algs(stats_gini,varname,"./plots/"+fprefix+"_ginihist_"+str(varname)+".pdf",yname="algorithm",trends=["contr_inv"],xlab="Population size",ylab="No. of contributions",font_size=16)
    plot_algs(stats_gini,varname,"./plots/"+fprefix+"_ginihistval_"+str(varname)+".pdf",yname="algorithm",trends=["val_inv"],xlab="Population size",ylab="Contribution values",font_size=16)
    plot_algs(stats_gini,varname,"./plots/"+fprefix+"_ginihistcost_"+str(varname)+".pdf",yname="algorithm",trends=["cost_inv"],xlab="Population size",ylab="Contribution costs",font_size=16)

## clean up memory
del res_decs,res_rews,res_percs,res_eval,rews_list,percs_list,decs_list,eval_list

for test,l in tests:
    res_eval=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("evaluation")])
    varnames=[v for v in ["N","n1","n2"] if (v in res_eval.columns) and (len(res_eval[v].unique())>1)]
    varvalues=expandgrid({v:res_eval[v].unique() for v in varnames})
    del res_eval
    print("reading qtables of "+str(test))
    ## compute qtable heatmaps
    vars2plot=["prob"]
    # var2plot="yes"
    try:
        qtables=pd.concat([pd.read_csv(os.path.join("./data/",str(test),f)) for f in os.listdir("./data/"+str(test)) if f.startswith("qtab")])
        # qtables=pd.read_csv("./data/"+str(test)+"/qtables.csv.gz")
        vars2plot=[v for v in vars2plot if v in qtables.columns]
        print("computing stats of qtabs")
        qtables_stats=compute_stats([qtables],idx=["state_val","state_cost"],columns=vars2plot)#+["num"])
        for var2plot in vars2plot:
            print("plotting var "+str(var2plot))
            plot_trend(qtables_stats,"state_cost","./plots/"+str(test)+"/qtables_"+str(var2plot)+"_cost.pdf",yname="state_val",trends=[var2plot])
            plot_trend(qtables_stats,"state_val","./plots/"+str(test)+"/qtables_"+str(var2plot)+"_val.pdf",yname="state_cost",trends=[var2plot])
        del qtables_stats
        ## compute histograms
        for var2plot in vars2plot:
            for idx,p in varvalues.iterrows():
                pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                print("plotting "+str(var2plot)+" histograms "+str(pdesc))
                q_exp=subset_df(qtables,p) # subset with current experimental conditions
                plot_qtable_hist(q_exp,"./plots/"+str(test)+"/qhist_"+pdesc+".pdf","state_val","state_cost",var2plot,str(dict(p)))
                prefix="./plots/"+str(test)+"/heatmaps"
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                plot_qtable_heat(q_exp,prefix+"/qheat_"+pdesc,"state_val","state_cost",var2plot)
                del q_exp
        print("stats_q")
        stats_q=compute_stats([qtables],idx=["state_val","state_cost"]+varnames,columns=vars2plot)#+["num"])
        for var2plot in vars2plot:
            for idx,p in varvalues.iterrows():
                print("plotting "+str(var2plot)+" qtabs "+str(dict(p)))
                pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                ## plot stats qtable
                try:
                    tmp=subset_df(stats_q,p)
                    f=lambda df,col: np.histogram2d(df["state_cost"],df["state_val"],weights=df[col+"_mean"],bins=[np.append(df["state_val"].unique(),[df["state_val"].max()+1]),np.append(df["state_cost"].unique(),[df["state_cost"].max()+1])])
                    heatmap_choice,xlabs,ylabs=f(tmp,var2plot)
                    plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_val_"+str(var2plot)+"_"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                    del heatmap_choice
                    heatmap_count,xlabs,ylabs=f(tmp,"num")
                    plot_hmap(heatmap_count,"Average number of occurrences of a state",str(test)+"/heat_q_count"+pdesc+".pdf","./plots",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                    del tmp#,heatmap_choice
                except Exception as e:
                    print("unable to plot heatmap for p: "+str(p))
                    print(e)
        del stats_q
        stats_qa=compute_stats([qtables],idx=["state_val","state_cost","idx"]+varnames,columns=vars2plot)#+["num"])
        for var2plot in vars2plot:
            for idx,p in varvalues.iterrows():
                print("plotting "+str(var2plot)+" individual qtabs "+str(dict(p)))
                pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                ## plot stats qtable
                try:
                    # plot individual qtables
                    tmp=subset_df(stats_qa,p)
                    prefix="./plots/"+str(test)+"/qtabs"
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)
                    for a in tmp["idx"].unique():
                        heatmap_choice,xlabs,ylabs=f(tmp[tmp["idx"]==a],var2plot)
                        plot_hmap(heatmap_choice,"Average qvalue associated to contribution","heat_q_val_"+str(var2plot)+"_"+str(a)+"_"+pdesc+".pdf",prefix,xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                    del tmp,heatmap_choice
                except Exception as e:
                    print("unable to plot heatmap for p: "+str(p))
                    print(e)
        del stats_qa
        del qtables

