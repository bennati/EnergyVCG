import argparse
import pandas as pd
import functools
from utils import *

parser = argparse.ArgumentParser(description='reads in parameters')

# Add the arguments for the parser to be passed on the cmd-line
# Defaults could be added default=
parser.add_argument('--indir', metavar='indir', nargs='?',default="data",help='the directory where to read the files')
parser.add_argument('--outdir', metavar='outdir', nargs='?',default="plots",help='the directory where to plot')
parser.add_argument('--losses',help='plot losses',action="store_true")
parser.add_argument('--by-param',help='plot aggregated results over each parameter configuration',action="store_true")
parser.add_argument('--by-time',help='plot aggregated results over time',action="store_true")
parser.add_argument('--qtabs',help='plot qtabs',action="store_true")
parser.add_argument('--comparison-indiv',help='plot individual comparison plots',action="store_true")
parser.add_argument('--comparison-norm',help='plot normalized comparison plots',action="store_true")
parser.add_argument('--individual-qtabs',help='plot individual qtabs',action="store_true")
parser.add_argument('--qtabs-histograms',help='plot histograms of qtabs',action="store_true")
args = parser.parse_args()

### TODOlist
## check that difference is gini is due to dataset (run sims with uniform data)
## check differences in perf using different values of n2 for nrel
## check that init of DQ has no patterns

datadir=args.indir
print("reading from "+str(datadir))
plotsdir=args.outdir
print("plotting in "+str(plotsdir))
baselines=["Rand","Full"]
fprefix="nrel"
tests=[["nrel_mand","Full","--"],["nrel_rand","Rand","--"],["nrel_knap","Cen.","-"],["nrel_asp","Asp.","-"]]
fprefix="nrelq"
tests=[["nrel_mand","Full","--"],["nrel_rand","Rand","--"],["nrel_knap","Cen.","-"],["nrel_Q","Q","-"]]
fprefix="unif"
tests=[["unif_mand","Full","--"],["unif_rand","Rand","--"],["unif_knap","Cen.","-"],["unif_asp","Asp.","-"]]
fprefix="unifq"
tests=[["unif_asp","Loc.","--"],["unif_rand","Rand","--"],["unif_knap","Cen.","--"],["unif_Q","Q","-"]]
# fprefix="gamma"
# tests=[["unif_Q",".001","-"],["unif_Q_a005",".005","-"],["unif_Q_a02",".02","-"],["unif_Q_a05",".05","-"]]
# fprefix="gammasup"
# tests=[["unif_Qsup",".001","-"],["unif_Qsup_a002",".002","-"],["unif_Qsup_a005",".005","-"],["unif_Qsup_a007",".007","-"]]
fprefix="ev"
tests=[["ev_mand","Baseline: Full","--"],["ev_rand","Baseline: Rnd","--"],["ev_knap","Optimization","-"],["ev_asp","Aspiration","-"],["ev_Q","Q-Learning","-"]]

def subset_df(df,conditions=pd.Series()):
    if conditions.empty:
        ret=df
    else:
        ret=df[functools.reduce(np.logical_and,[(df[k]==v) for k,v in zip(conditions.index,conditions)])] # select only a subset of the table matching the parameters
        ret.reset_index()
    return ret

def plot_algs(df,xname,filename,trends=None,yname=None,lstyles=['-','--',':','-.'],colors=None,font_size=12,ylab=None,xlab=None,baselines=[]):
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
    colorArtists = []
    baselineArtists = []
    lines=[]
    for (d,l),c in zip(data,colors):
        x=d[xname]
        sty=d["lsty"].unique()[0]
        art=plt.Line2D((0,1),(0,0), color=c, marker='', linestyle=sty,linewidth=3)
        if l in baselines:
            baselineArtists.append(art)
        else:
            colorArtists.append(art)
        for y in trends:
            lab=(y if l is None else y+"; "+yname+"="+str(l))
            lines.append({"x":x,"y":d[y+"_mean"],"ci":d[y+"_ci"],"label":lab,"linestyle":sty,"color":c})
    box = ax.get_position()
    # if len(baselineArtists)==0:
    #     ax.set_position([box.x0, box.y0,box.width, box.height * 1.1])
    #     ax.add_artist(plt.legend(colorArtists,[l for d,l in data if l not in baselines],loc='upper center', bbox_to_anchor=(0.5, 1.18),fancybox=True, shadow=True, ncol=len(data),fontsize=font_size))
    # else:
    #     ax.set_position([box.x0, box.y0,box.width, box.height * 1.2])
    #     ax.add_artist(plt.legend(baselineArtists,[l for d,l in data if l in baselines],loc='upper center', bbox_to_anchor=(0.5, 1.18),fancybox=True, shadow=True, ncol=len(data),fontsize=font_size))
    #     ax.add_artist(plt.legend(colorArtists,[l for d,l in data if l not in baselines],loc='upper center', bbox_to_anchor=(0.5, 1.33),fancybox=True, shadow=True, ncol=len(data),fontsize=font_size))
    for l in lines:
        ax.plot(l["x"],l["y"],label=l["label"],linestyle=l["linestyle"],color=l["color"],linewidth=3)
        ax.fill_between(l["x"],np.asarray(l["y"])-np.asarray(l["ci"]),np.asarray(l["y"])+np.asarray(l["ci"]),alpha=0.2,linestyle=l["linestyle"],facecolor=l["color"],linewidth=3)
    # plt.legend()
    plt.setp(ax.xaxis.get_majorticklabels(),fontsize=font_size)
    plt.setp(ax.yaxis.get_majorticklabels(),fontsize=font_size)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

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
    if not os.path.exists("./"+plotsdir+"/"+str(test)):
        os.makedirs("./"+plotsdir+"/"+str(test))
    res_decs=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("decisions")])
    # res_decs=pd.read_csv("./"+datadir+"/"+str(test)+"/decisions.csv.gz")
    res_decs["algorithm"]=l
    res_decs["lsty"]=sty
    res_decs["cost_pop"]=[(0 if np.isnan(i) else i)  for i in res_decs["cost"]]
    decs_list.append(res_decs)
    res_eval=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("evaluation")])
    # res_eval=pd.read_csv("./"+datadir+"/"+str(test)+"/evaluation.csv.gz")
    res_eval["algorithm"]=l
    res_eval["lsty"]=sty
    eval_list.append(res_eval)
    varnames=[v for v in ["N","n1","n2"] if (v in res_eval.columns) and (len(res_eval[v].unique())>1)]
    varvalues=expandgrid({v:res_eval[v].unique() for v in varnames})
    try:
        contrib_hist=pd.read_csv("./"+datadir+"/"+str(test)+"/contrib_hist.csv.gz")
    except:
        contrib_hist=None
    # varnames=[c for c in contrib_hist.columns if c not in ["value","cnt"]]
    # varvalues=expandgrid({v:contrib_hist[v].unique() for v in varnames})
    try:
        stats_gini_contribs=pd.read_csv("./"+datadir+"/"+str(test)+"/stats_gini_contribs.csv.gz")
    except:
        stats_gini_contribs=res_decs.groupby(varnames+["agentID","repetition"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum,"cost":np.sum}) # sum up all contributions in each simulation (over all timesteps)
        stats_gini_contribs=stats_gini_contribs.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini,"cost":gini}) # compute gini coefficient across agents
        stats_gini_contribs=stats_gini_contribs.rename(columns={"contributed":"contributed_hist","contribution":"contribution_hist","cost":"cost_hist"})
    stats_gini_contribs["algorithm"]=l
    stats_gini_contribs["lsty"]=sty
    gini_contribs.append(stats_gini_contribs)
    if args.losses:
        try:
            losses=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("loss")])
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
            fig.savefig("./"+plotsdir+"/"+str(test)+"/loss.pdf",format='pdf')
            plt.close(fig)
            # plot_trend(losses,"bins","./"+plotsdir+"/"+str(test)+"/loss.pdf",trends=cols)
        except Exception as e:
            print("cannot print losses "+str(e))
            losses=None
    print("done reading files")
    ### now move to computing statistics that aggregate on one of the parameters ###
    if args.by_param:
        res_percs=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("perception")])
        # res_percs=pd.read_csv("./"+datadir+"/"+str(test)+"/perceptions.csv.gz")
        res_percs["algorithm"]=l
        res_decs["lsty"]=sty
        percs_list.append(res_percs)
        res_rews=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("reward")])
        # res_rews=pd.read_csv("./"+datadir+"/"+str(test)+"/rewards.csv.gz")
        res_rews["algorithm"]=l
        res_decs["lsty"]=sty
        rews_list.append(res_rews)
        for varname in varnames:
            print("plotting stats for var "+str(varname))
            stats_gini=compute_stats([stats_gini_contribs],[varname],columns=["contributed_hist","contribution_hist"]) # average across repetitions
            plot_trend(stats_gini,varname,"./"+plotsdir+"/"+str(test)+"/gini_"+str(varname)+".pdf")
            stats_rews=compute_stats(res_rews,idx=[varname],columns=["reward"])
            plot_trend(stats_rews,varname,"./"+plotsdir+"/"+str(test)+"/rewards_"+str(varname)+".pdf")
            f=functools.partial(subset_df,conditions=pd.Series({"timestep":int(res_decs["timestep"].max())}))
            stats_percs=compute_stats(f(res_percs),idx=[varname],columns=["value","cost"])
            plot_trend(stats_percs,varname,"./"+plotsdir+"/"+str(test)+"/perceptions_"+str(varname)+".pdf")
            stats_decs=compute_stats(f(res_decs),idx=[varname],columns=["contribution","cost","cost_pop","contributed"])
            plot_trend(stats_decs,varname,"./"+plotsdir+"/"+str(test)+"/decisions_"+str(varname)+".pdf")
            stats_eval=compute_stats(f(res_eval),idx=[varname],columns=["gini","gini_cost","cost_pop","efficiency","social_welfare","success","num_contrib"])
            plot_measures(stats_eval,varname,"./"+plotsdir+"/"+str(test)+"/eval_"+str(varname)+".pdf")
            if contrib_hist is not None:
                stats_contrib_hist=compute_stats(contrib_hist,idx=[varname,"value"],columns=["cnt"])
                plot_trend(stats_contrib_hist,"value","./"+plotsdir+"/"+str(test)+"/contrib_hist_"+str(varname)+".pdf",yname=varname)

    ### now compute statistics for each parameter configuration, aggregating only on repetitions ###
    if args.by_time:
        stats_evalt=compute_stats(res_eval,idx=["timestep"],columns=["gini","gini_cost","cost_pop","efficiency","social_welfare","success","num_contrib"])
        plot_measures(stats_evalt,"timestep","./"+plotsdir+"/"+str(test)+"/eval_"+str("time")+".pdf")
        print("plotting individual params")
        stats_t=compute_stats(res_eval,idx=["timestep"]+varnames,columns=["gini","gini_cost","cost_pop","efficiency","social_welfare","success","num_contrib"])
        for idx,p in varvalues.iterrows():
            print("plotting params "+str(dict(p)))
            pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
            # temporal evolution of measures
            tmp=subset_df(stats_t,p)
            plot_measures(tmp,"timestep","./"+plotsdir+"/"+str(test)+"/time_"+pdesc+".pdf")

### generate comparison plots
print("generating comparison plots at times "+str([int(d["timestep"].max()) for d in decs_list]))
decs_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in decs_list]).drop_duplicates()
decs_list["privacy_inv"]=1-decs_list["privacy"]
eval_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in eval_list]).drop_duplicates()
eval_list["gini_inv"]=1-eval_list["gini"]
eval_list["gini_cost_inv"]=1-eval_list["gini_cost"]
gini_contribs=pd.concat(gini_contribs)
gini_contribs["contributed_hist_inv"]=1-gini_contribs["contributed_hist"]
gini_contribs["contribution_hist_inv"]=1-gini_contribs["contribution_hist"]
gini_contribs["contributed_inv"]=1-gini_contribs["contributed"]
gini_contribs["contribution_inv"]=1-gini_contribs["contribution"]
## boxplots
def boxplot(df,measure,label):
    data=compute_stats([df],idx=["algorithm"]+varnames,columns=[measure])
    data=data.groupby("algorithm").apply(lambda x: np.array(x[measure+"_mean"]))
    # data=data[["Q-Learning","Aspiration","Optimization","Baseline: Rnd","Baseline: Full"]] # sort
    fig, ax = plt.subplots()
    ax.boxplot(data,labels=data.keys(),showfliers=False,vert=False)
    font_size=16
    ax.set_xlabel(label,fontsize=font_size)
    plt.setp(ax.xaxis.get_majorticklabels(),fontsize=font_size)
    plt.setp(ax.yaxis.get_majorticklabels(),fontsize=font_size)
    fig.savefig("./"+plotsdir+"/"+str(test)+"_boxplot_"+measure+".pdf",format='pdf')
bardata=pd.DataFrame()
for df,measure,label in [[decs_list,"privacy_inv","Privacy"],[eval_list,"efficiency","Efficiency"],[eval_list,"social_welfare","Welfare"],[gini_contribs,"contributed_inv","Fairness of contributions"]]:
    # boxplot(df,measure,label)
    tmp=compute_stats([df],idx=["algorithm"],columns=[measure])
    if bardata.empty:
        bardata=tmp
    else:
        bardata=pd.merge(bardata,tmp,on='algorithm',how='outer')
## bar charts
# for groups,labels in zip([['efficiency'],['social_welfare'],['contributed_inv'],['privacy_inv']],[["Efficiency"],["Welfare"],["Fairness of contribution"],["Privacy"]]):
#     varnames=["Baseline: Rnd","Baseline: Full","Optimization","Aspiration","Q-Learning"] #bardata['algorithm']
#     n_vars=len(varnames)
#     fig, ax = plt.subplots(figsize=(3.5,5))
#     index = np.arange(len(groups))
#     bar_width = 1/(n_vars+1)
#     offsets=np.arange(0,1,bar_width)
#     opacity = 0.8
#     rects=[]
#     cx1 = plt.get_cmap('cubehelix_r')
#     cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=n_vars+1)
#     scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cx1)
#     colors=[scalarMap.to_rgba(i) for i in range(1,n_vars+1)]
#     for i,c in enumerate(colors):
#         alg=varnames[i]
#         cols=np.asarray(bardata[bardata['algorithm']==alg][[g+"_mean" for g in groups]])[0]
#         cis=np.asarray(bardata[bardata['algorithm']==alg][[g+"_ci" for g in groups]])[0]
#         rects.append(plt.bar(index+offsets[i], cols, yerr=cis, width=bar_width,
#                              alpha=opacity,
#                              color=c,
#                              label=alg,
#                              capsize=5))
#     # plt.xlabel("Contribution strategy")
#     # plt.ylabel()
#     plt.title(labels[0])
#     # plt.xticks(index + offsets.mean(), labels)
#     plt.xticks([])
#     plt.legend(loc=4)
#     plt.tight_layout()
#     fig.savefig("./"+plotsdir+"/barchart_"+str(groups)+".pdf",format='pdf')
for varname in varnames:
    print("plotting var "+str(varname))
    xlab=("Population size" if varname=="N" else "Action-state space size")
    stats_decs=compute_stats([decs_list],idx=[varname,"algorithm","lsty"],columns=["contribution","cost","cost_pop","contributed","privacy_inv"])
    # plot_algs(stats_decs,varname,"./"+plotsdir+"/"+fprefix+"_costs_volunteers_"+str(varname)+".pdf",yname="algorithm",trends=["cost"],xlab=xlab,ylab="Cost for volunteers",font_size=16)
    # plot_algs(stats_decs,varname,"./"+plotsdir+"/"+fprefix+"_costs_global_"+str(varname)+".pdf",yname="algorithm",trends=["cost_pop"],xlab=xlab,ylab="Cost",font_size=16)
    plot_algs(stats_decs,varname,"./"+plotsdir+"/"+fprefix+"_priv_inv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy_inv"],xlab=xlab,ylab="Privacy",font_size=16,baselines=baselines)
    stats_eval=compute_stats([eval_list],idx=[varname,"algorithm","lsty"],columns=["gini_inv","gini_cost_inv","cost_pop","efficiency","social_welfare","success","num_contrib"])
    # plot_algs(stats_eval,varname,"./"+plotsdir+"/"+fprefix+"_gini_"+str(varname)+".pdf",yname="algorithm",trends=["gini_inv"],xlab=xlab,ylab="Fairness of contributions",font_size=16)
    # plot_algs(stats_eval,varname,"./"+plotsdir+"/"+fprefix+"_ginicost_"+str(varname)+".pdf",yname="algorithm",trends=["gini_cost_inv"],xlab=xlab,ylab="Fairness of costs",font_size=16)
    plot_algs(stats_eval,varname,"./"+plotsdir+"/"+fprefix+"_success_"+str(varname)+".pdf",yname="algorithm",trends=["success"],xlab=xlab,ylab="Success rate",font_size=16,baselines=baselines)
    plot_algs(stats_eval,varname,"./"+plotsdir+"/"+fprefix+"_welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"],xlab=xlab,ylab="Social welfare",font_size=16,baselines=baselines)
    plot_algs(stats_eval,varname,"./"+plotsdir+"/"+fprefix+"_eff_"+str(varname)+".pdf",yname="algorithm",trends=["efficiency"],xlab=xlab,ylab="Efficiency",font_size=16,baselines=baselines)
    # plot_algs(stats_eval,varname,"./"+plotsdir+"/priv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy"])
    stats_gini=compute_stats([gini_contribs],[varname,"algorithm","lsty"],columns=["contributed_inv","contribution_inv","contributed_hist_inv","contribution_hist_inv"]) # average across repetitions
    # stats_gini=compute_stats([gini_contribs],[varname,"algorithm","lsty","cost"],columns=["contributed_hist","contribution_hist"]) # average across repetitions
    # stats_gini["cost_inv_mean"]=1-stats_gini["cost_mean"]
    # stats_gini["cost_inv_ci"]=stats_gini["cost_ci"]
    plot_algs(stats_gini,varname,"./"+plotsdir+"/"+fprefix+"_ginibool_"+str(varname)+".pdf",yname="algorithm",trends=["contributed_inv"],xlab=xlab,ylab="Fairness of contributions",font_size=16,baselines=baselines)
    plot_algs(stats_gini,varname,"./"+plotsdir+"/"+fprefix+"_ginihist_"+str(varname)+".pdf",yname="algorithm",trends=["contributed_hist_inv"],xlab=xlab,ylab="Fairness of contributions, over time",font_size=16,baselines=baselines)
    # plot_algs(stats_gini,varname,"./"+plotsdir+"/"+fprefix+"_ginihistval_"+str(varname)+".pdf",yname="algorithm",trends=["contribution_hist_inv"],xlab=xlab,ylab="Fairness of contributions, over time",font_size=16)
    # plot_algs(stats_gini,varname,"./"+plotsdir+"/"+fprefix+"_ginihistcost_"+str(varname)+".pdf",yname="algorithm",trends=["cost_inv"],xlab=xlab,ylab="Contribution costs",font_size=16)
    if args.comparison_indiv:
        print("plotting individual params")
        stats_gini2=compute_stats([gini_contribs],varnames+["algorithm","lsty"],columns=["contributed_hist_inv","contribution_hist_inv"]) # average across repetitions
        for varname in varnames:
            othervars=[v for v in varnames if v!=varname]
            for othervar in othervars:
                print(othervar)
                for v in varvalues[varname].unique():
                    print(v)
                    tmp=stats_gini2[stats_gini2[varname]==v]
                    fname=varname+str(v)+"_"+othervar+".pdf"
                    plot_algs(tmp,othervar,"./"+plotsdir+"/"+fprefix+"_ginihist_"+fname,yname="algorithm",trends=["contributed_hist_inv"],xlab=xlab,ylab="Fairness of contributions",font_size=16)
                    plot_algs(tmp,othervar,"./"+plotsdir+"/"+fprefix+"_ginihistval_"+fname,yname="algorithm",trends=["contribution_hist_inv"],xlab=xlab,ylab="Fairness of contributions, over time",font_size=16)

if args.comparison_norm:
    ### compute normalized measures on baseline Full
    def body_normalize(df1,norm,cols):
        ret=df1.copy()
        keeps=[c for c in norm.columns if c in varnames+["repetition","agentID"]]
        qqq=pd.merge(norm,df1,on=keeps,how="inner")
        for v in cols:
            ret[v]=qqq[v+"_y"]/qqq[v+"_x"]
            # ret[v]=pd.merge(ret,qqq,on=keeps,how="inner")[v]
            ret.drop(ret[ret[v]==np.inf].index,inplace=True)
        return ret
    def normalize_measures(df,ref,algs,cols):
        return df[df["algorithm"].isin(algs)].groupby(["algorithm"]).apply(lambda x: body_normalize(x,df[df["algorithm"]==ref],cols))
    for varname in varnames:
        print("plotting var "+str(varname))
        xlab=("Population size" if varname=="N" else "Action-state space size")
        # decs_norm=normalize_measures(decs_list,ref,algs,cols=["contribution","cost","cost_pop","contributed","privacy"])
        # stats_decs_norm=compute_stats([decs_norm],idx=[varname,"algorithm","lsty"],columns=["contribution","cost","cost_pop","contributed","privacy"])
        # stats_decs_norm["privacy_inv_mean"]=1-stats_decs_norm["privacy_mean"]
        # stats_decs_norm["privacy_inv_ci"]=stats_decs_norm["privacy_ci"]
        # plot_algs(stats_decs_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_costs_volunteers_"+str(varname)+".pdf",yname="algorithm",trends=["cost"],xlab=xlab,ylab="Cost for volunteers",font_size=16)
        # plot_algs(stats_decs_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_costs_global_"+str(varname)+".pdf",yname="algorithm",trends=["cost_pop"],xlab=xlab,ylab="Cost",font_size=16)
        # plot_algs(stats_decs_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_priv_inv_"+str(varname)+".pdf",yname="algorithm",trends=["privacy_inv"],xlab=xlab,ylab="Privacy",font_size=16)
        ref="Full"
        algs=[b for a,b,c in tests if b!=ref]
        eval_norm=normalize_measures(eval_list,ref,algs,cols=["gini_inv"])
        stats_eval_norm=compute_stats([eval_norm],idx=[varname,"algorithm","lsty"],columns=["gini_inv","social_welfare"])
        plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_gini_"+str(varname)+".pdf",yname="algorithm",trends=["gini_inv"],xlab=xlab,ylab="Fairness of contributions, normalized",font_size=16)
        # stats_eval_norm["gini_cost_inv_mean"]=1-stats_eval_norm["gini_cost_mean"]
        # stats_eval_norm["gini_cost_inv_ci"]=stats_eval_norm["gini_cost_ci"]
        # plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_ginicost_"+str(varname)+".pdf",yname="algorithm",trends=["gini_cost_inv"],xlab=xlab,ylab="Fairness of costs",font_size=16)
        # plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_success_"+str(varname)+".pdf",yname="algorithm",trends=["success"],xlab=xlab,ylab="Success rate",font_size=16)
        # plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"],xlab=xlab,ylab="Social welfare",font_size=16)
        ref="Cen."
        algs=[b for a,b,c in tests if b!=ref]
        eval_norm=normalize_measures(eval_list,ref,algs,cols=["efficiency","social_welfare"])
        stats_eval_norm=compute_stats([eval_norm],idx=[varname,"algorithm","lsty"],columns=["efficiency","social_welfare"])
        plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_eff_"+str(varname)+".pdf",yname="algorithm",trends=["efficiency"],xlab=xlab,ylab="Efficiency, normalized",font_size=16)
        plot_algs(stats_eval_norm,varname,"./"+plotsdir+"/"+fprefix+"_norm_welfare_"+str(varname)+".pdf",yname="algorithm",trends=["social_welfare"],xlab=xlab,ylab="Social welfare, normalized",font_size=16)

try:
    percs_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in percs_list])
    rews_list=pd.concat([subset_df(d,conditions=pd.Series({"timestep":d["timestep"].max()})) for d in rews_list])
    for varname in varnames:
        print("plotting var "+str(varname))
        stats_rews=compute_stats([rews_list],idx=[varname,"algorithm"],columns=["reward"])
        plot_algs(stats_rews,varname,"./"+plotsdir+"/wtest_rewards_"+str(varname)+".pdf",yname="algorithm",trends=["reward"],xlab=xlab,ylab="Reward",font_size=16)
        stats_percs=compute_stats([perc_list],idx=[varname,"algorithm"],columns=["value","cost"])
        plot_algs(stats_percs,varname,"./"+plotsdir+"/"+str(test)+"/perceptions_"+str(varname)+".pdf",yname="algorithm")
except:
    pass

## clean up memory
del res_decs,res_rews,res_percs,res_eval,rews_list,percs_list,decs_list,eval_list

if args.qtabs:
    for test,l,sty in tests:
        res_eval=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("evaluation")])
        varnames=[v for v in ["N","n1","n2"] if (v in res_eval.columns) and (len(res_eval[v].unique())>1)]
        varvalues=expandgrid({v:res_eval[v].unique() for v in varnames})
        del res_eval
        print("reading qtables of "+str(test))
        ## compute qtable heatmaps
        vars2plot=["prob","yes"]
        # var2plot="yes"
        try:
            qtables=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("qtab")])
            # qtables=pd.read_csv("./"+datadir+"/"+str(test)+"/qtables.csv.gz")
            vars2plot=[v for v in vars2plot if v in qtables.columns]
            print("computing stats of qtabs")
            qtables_stats=compute_stats([qtables],idx=["state_val","state_cost"],columns=vars2plot)#+["num"])
            for var2plot in vars2plot:
                print("plotting var "+str(var2plot))
                plot_trend(qtables_stats,"state_cost","./"+plotsdir+"/"+str(test)+"/qtables_"+str(var2plot)+"_cost.pdf",yname="state_val",trends=[var2plot])
                plot_trend(qtables_stats,"state_val","./"+plotsdir+"/"+str(test)+"/qtables_"+str(var2plot)+"_val.pdf",yname="state_cost",trends=[var2plot])
            del qtables_stats
            ## compute histograms
            if args.qtabs_histograms:
                for var2plot in vars2plot:
                    for idx,p in varvalues.iterrows():
                        pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                        print("plotting "+str(var2plot)+" histograms "+str(pdesc))
                        q_exp=subset_df(qtables,p) # subset with current experimental conditions
                        plot_qtable_hist(q_exp,"./"+plotsdir+"/"+str(test)+"/qhist_"+pdesc+".pdf","state_val","state_cost",var2plot,str(dict(p)))
                        prefix="./"+plotsdir+"/"+str(test)+"/heatmaps"
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
                        plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_val_"+str(var2plot)+"_"+pdesc+".pdf","./"+plotsdir+"/",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                        del heatmap_choice
                        # heatmap_count,xlabs,ylabs=f(tmp,"num")
                        # plot_hmap(heatmap_count,"Average number of occurrences of a state",str(test)+"/heat_q_count"+pdesc+".pdf","./"+plotsdir+"/",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
                        del tmp#,heatmap_choice
                    except Exception as e:
                        print("unable to plot heatmap for p: "+str(p))
                        print(e)
            del stats_q
            if args.individual_qtabs:
                stats_qa=compute_stats([qtables],idx=["state_val","state_cost","idx"]+varnames,columns=vars2plot)#+["num"])
                for var2plot in vars2plot:
                    for idx,p in varvalues.iterrows():
                        print("plotting "+str(var2plot)+" individual qtabs "+str(dict(p)))
                        pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
                        ## plot stats qtable
                        try:
                            # plot individual qtables
                            tmp=subset_df(stats_qa,p)
                            prefix="./"+plotsdir+"/"+str(test)+"/qtabs"
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
            ###
        #     print("computing stats of qtabs, history")
        #     qtables=pd.concat([pd.read_csv(os.path.join("./"+datadir+"/",str(test),f)) for f in os.listdir("./"+datadir+"/"+str(test)) if f.startswith("qtab")])
        #     # qtables=pd.read_csv("./"+datadir+"/"+str(test)+"/qtables.csv.gz")
        #     vars2plot=[v for v in vars2plot if v in qtables.columns]
        #     print("computing stats of qtabs")
        #     qtables_stats=compute_stats([qtables],idx=["hist_contributed","avg_hist_contributed"],columns=vars2plot)#+["num"])
        #     for var2plot in vars2plot:
        #         print("plotting var "+str(var2plot))
        #         plot_trend(qtables_stats,"avg_hist_contributed","./"+plotsdir+"/"+str(test)+"/qtables_"+str(var2plot)+"_cost.pdf",yname="hist_contributed",trends=[var2plot])
        #         plot_trend(qtables_stats,"hist_contributed","./"+plotsdir+"/"+str(test)+"/qtables_"+str(var2plot)+"_hist.pdf",yname="avg_hist_contributed",trends=[var2plot])
        #     del qtables_stats
        #     ## compute histograms
        #     # for var2plot in vars2plot:
        #     #     for idx,p in varvalues.iterrows():
        #     #         pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        #     #         print("plotting "+str(var2plot)+" histograms "+str(pdesc))
        #     #         q_exp=subset_df(qtables,p) # subset with current experimental conditions
        #     #         plot_qtable_hist(q_exp,"./"+plotsdir+"/"+str(test)+"/qhist_"+pdesc+".pdf","hist_contributed","avg_hist_contributed",var2plot,str(dict(p)))
        #     #         prefix="./"+plotsdir+"/"+str(test)+"/heatmaps"
        #     #         if not os.path.exists(prefix):
        #     #             os.makedirs(prefix)
        #     #         plot_qtable_heat(q_exp,prefix+"/qheat_"+pdesc,"hist_contributed","avg_hist_contributed",var2plot)
        #     #         del q_exp
        #     print("stats_q")
        #     stats_q=compute_stats([qtables],idx=["hist_contributed","avg_hist_contributed"]+varnames,columns=vars2plot)#+["num"])
        #     for var2plot in vars2plot:
        #         for idx,p in varvalues.iterrows():
        #             print("plotting "+str(var2plot)+" qtabs "+str(dict(p)))
        #             pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        #             ## plot stats qtable
        #             try:
        #                 tmp=subset_df(stats_q,p)
        #                 f=lambda df,col: np.histogram2d(df["avg_hist_contributed"],df["hist_contributed"],weights=df[col+"_mean"],bins=[np.append(df["hist_contributed"].unique(),[df["hist_contributed"].max()+1]),np.append(df["avg_hist_contributed"].unique(),[df["avg_hist_contributed"].max()+1])])
        #                 heatmap_choice,xlabs,ylabs=f(tmp,var2plot)
        #                 plot_hmap(heatmap_choice,"Average qvalue associated to contribution",str(test)+"/heat_q_hist_"+str(var2plot)+"_"+pdesc+".pdf","./"+plotsdir+"/",xlab="Own",ylab="Avg",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
        #                 del heatmap_choice
        #                 # heatmap_count,xlabs,ylabs=f(tmp,"num")
        #                 # plot_hmap(heatmap_count,"Average number of occurrences of a state",str(test)+"/heat_q_count"+pdesc+".pdf","./"+plotsdir+"/",xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
        #                 del tmp#,heatmap_choice
        #             except Exception as e:
        #                 print("unable to plot heatmap for p: "+str(p))
        #                 print(e)
        #     del stats_q
        #     # stats_qa=compute_stats([qtables],idx=["hist_contributed","avg_hist_contributed","idx"]+varnames,columns=vars2plot)#+["num"])
        #     # for var2plot in vars2plot:
        #     #     for idx,p in varvalues.iterrows():
        #     #         print("plotting "+str(var2plot)+" individual qtabs "+str(dict(p)))
        #     #         pdesc="_".join([str(k)+str(v) for k,v in dict(p).items()])
        #     #         ## plot stats qtable
        #     #         try:
        #     #             # plot individual qtables
        #     #             tmp=subset_df(stats_qa,p)
        #     #             prefix="./"+plotsdir+"/"+str(test)+"/qtabs"
        #     #             if not os.path.exists(prefix):
        #     #                 os.makedirs(prefix)
        #     #             for a in tmp["idx"].unique():
        #     #                 heatmap_choice,xlabs,ylabs=f(tmp[tmp["idx"]==a],var2plot)
        #     #                 plot_hmap(heatmap_choice,"Average qvalue associated to contribution","heat_q_hist_"+str(var2plot)+"_"+str(a)+"_"+pdesc+".pdf",prefix,xlab="Value",ylab="Cost",ticks=[range(len(xlabs[:-1])),range(len(ylabs[:-1]))],ticklabs=[xlabs[:-1],ylabs[:-1]],inverty=False)
        #     #             del tmp,heatmap_choice
        #     #         except Exception as e:
        #     #             print("unable to plot heatmap for p: "+str(p))
        #     #             print(e)
        #     # del stats_qa
        #     del qtables
        except Exception as e:
            print("qtables not found: "+str(e))
