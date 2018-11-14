import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})

def plot_trend(df,xname,filename,trends=None,yname=None,
               lstyles=['-','--',':','-.'],colors=None,font_size=4,ylab=None,xlab=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    if len(trends)>len(lstyles):
        lstyles=['-']*len(trends)
    else:
        lstyles=lstyles[:len(trends)]
    lineArtists=[plt.Line2D((0,1),(0,0), color='k', marker='', linestyle=sty) for sty in lstyles]
    ax.set_xlabel(xlab or xname)
    ax.set_ylabel(ylab or yname)
    if yname is None:
        data=[(df,None)]
    else:
        data=[(df[df[yname]==i],i) for i in df[yname].unique()]
    if colors is None:
        cmap = plt.get_cmap('cubehelix_r')
        colors=[cmap(float(i+1)/(len(data)+1)) for i in range(len(data))]
    colorArtists = [plt.Line2D((0,1),(0,0), color=c) for c in colors]
    #fig.suptitle(title)
    # if ylim:
    #     ax.set_ylim(ylim)
    box = ax.get_position()
    # if yname and len(yname)>1 and len(trends)>1:
    #     # Shrink current axis's height by 10% on the bottom
    #     ax.set_position([box.x0, box.y0 + box.height * 0.2,
    #                   box.width, box.height * 0.8])
    if yname and len(df[yname].unique())>1:
        ax.add_artist(plt.legend(colorArtists,[l for d,l in data],
                                 loc='upper center', bbox_to_anchor=(0.5, 1.11),fancybox=True,
                                 shadow=True, ncol=len(data),fontsize=font_size))
    if len(trends)>1:
        ax.add_artist(plt.legend(lineArtists,trends,loc='upper center',
                                 bbox_to_anchor=(0.5, 1),fancybox=True, shadow=True,
                                 ncol=len(trends),fontsize=font_size))
    for (d,l),c in zip(data,colors):
        x=d[xname]
        for y,sty in zip(trends,lstyles):
            lab=(y if l is None else y+"; "+yname+"="+str(l))
            ax.plot(x,d[y+"_mean"],label=lab,linestyle=sty,color=c)
            ax.fill_between(x,np.asarray(d[y+"_mean"])-np.asarray(d[y+"_ci"]),
                            np.asarray(d[y+"_mean"])+np.asarray(d[y+"_ci"]),
                            alpha=0.2,linestyle=sty,facecolor=c)
    # plt.legend()
    fig.savefig(filename,format='png')
    plt.close(fig)

datadir="./output/"
plotdir=os.path.join(datadir,"plots_cmp")
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

#Exp 1
treatments=[["exp_base_10_0025","Bilateral"],
            ["exp4_10_0025","Multibid"]]
varnames=[["bias_high","Discrimination"]]
# Exp 2
treatments=[["exp_base_10_0025","Bilateral"],
            ["exp1_10_0025","Mediated"],
            ["exp4_10_0025","Multibid"]]
varnames=[[["bias_high","bias_mediator"],"Discrimination"],["consumption_offset","Consumption offset"]]
# Exp 3
treatments=[["exp_base_10_0025","Bilateral"],
            ["exp1_10_0025","Mediated"],
            ["exp3_10_0025","Mediated bidsplitting"]]
varnames=[[["bias_high","bias_mediator"],"Discrimination"]]

measures_eval=[["efficiency","Efficiency"],
               ["satifaction_cons_low","theta low"],
               ["satifaction_cons_high","theta high"],
               ["satifaction_prod_low","eta low"],
               ["satifaction_prod_high","eta high"]]

measures_decs=[] #[["cost","Cost"]]
measures_percs=[["production","Rest production"],["consumption","Rest consumption"],["tariff","Tariff"]]
measures_rews=[] #[["reward","Reward"]]

def read_data_type(t,v):
    ret=[]
    if not isinstance(v,list):
        v_=[v]
    else:
        v_=v
    for x in v_:                 # if there is more than one index
        for d,l in treatments:
            fname=os.path.join(datadir,d,"agg_"+t+"_"+x+".csv")
            if os.path.isfile(fname):
                tmp=pd.read_csv(fname)
                tmp["treatment"]=l
                ret.append(tmp)
    if ret!=[]:
        ret=pd.concat(ret)
        if isinstance(v,list):
            ret['index']=ret[v[0]]
            for x in v[1:]:
                ret['index']=ret['index'].add(ret[x],fill_value=0)
    else:
        ret=pd.DataFrame()
    return ret
for v,vl in varnames:
    x=('index' if isinstance(v,list) else v)
    evaluations=read_data_type("evaluations",v)
    decisions=read_data_type("decisions",v)
    perceptions=read_data_type("perceptions",v)
    rewards=read_data_type("rewards",v)
    if not evaluations.empty:
        for m,l in measures_eval:
            plot_trend(evaluations,x,os.path.join(plotdir,"eval_"+str(v)+"_"+str(m)+".png"),
                       yname="treatment",trends=[m],font_size=12,ylab=l,xlab=vl)
    if not decisions.empty:
        for m,l in measures_decs:
            plot_trend(decisions,x,os.path.join(plotdir,"decs_"+str(v)+"_"+str(m)+".png"),
                       yname="treatment",trends=[m],font_size=12,ylab=l,xlab=vl)
    if not perceptions.empty:
        for m,l in measures_percs:
            plot_trend(perceptions,x,os.path.join(plotdir,"percs_"+str(v)+"_"+str(m)+".png"),
                       yname="treatment",trends=[m],font_size=12,ylab=l,xlab=vl)
    if not rewards.empty:
        for m,l in measures_rews:
            plot_trend(rewards,x,os.path.join(plotdir,"rews_"+str(v)+"_"+str(m)+".png"),
                       yname="treatment",trends=[m],font_size=12,ylab=l,xlab=vl)
