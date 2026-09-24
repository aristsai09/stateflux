import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib import gridspec
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from PIL import Image

# Resolve all inputs relative to the repository so this script is portable.
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('STATEFLUX_FIGURE_OUT', REPO_ROOT / 'figures'))
OUT.mkdir(parents=True, exist_ok=True)

DILI_FOLD = REPO_ROOT / 'data/dili/dili_postrepair_fold_metrics.csv'
DILI_METRICS = REPO_ROOT / 'data/dili/dili_postrepair_metrics.csv'
EXTERNAL = REPO_ROOT / 'data/external/external_postrepair_metrics.csv'
BOOT = REPO_ROOT / 'data/dili/paired_bootstrap_postrepair_auroc_exact_rng.json'
HAGAN_PRED = REPO_ROOT / 'data/genotoxicity/hagan_lowdim_predictions.csv'
HAGAN_FOLD = REPO_ROOT / 'data/genotoxicity/hagan_lowdim_fold_metrics.csv'
HAGAN_MET = REPO_ROOT / 'data/genotoxicity/hagan_lowdim_metrics.csv'
DFT24 = REPO_ROOT / 'data/dft/dft_24_edge_validation.csv'
DFTCLASS = REPO_ROOT / 'data/dft/dft_delta_omega_by_class.csv'

# restrained ACS-like palette
BLACK = '#222222'
MID = '#777777'
LIGHT = '#D5D5D5'
PALE = '#F3F3F3'
CRIMSON = '#8C1D40'
TEAL = '#247B7B'
PURPLE = '#6A4C93'
BLUEGRAY = '#5D6D7E'
MODELS = ['ECFP','ECFP_StateCore','ECFP_FluxCore','ECFP_StateFluxCore']
MODEL_LABELS = ['ECFP','+ StateCore','+ FluxCore','+ StateFluxCore']
MODEL_COLORS = [BLACK, '#9A9A9A', TEAL, PURPLE]
CLASS_ORDER = ['dealkylation','oxidation','phaseII_conjugation']
CLASS_LABELS = ['Dealkylation','Oxidation','Phase II']
CLASS_COLORS = [BLACK, CRIMSON, TEAL]
CLASS_MARKERS = ['o','s','^']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.4,
    'axes.labelsize': 7.8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 6.8,
    'ytick.labelsize': 6.8,
    'legend.fontsize': 6.8,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.65,
    'ytick.major.width': 0.65,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'svg.fonttype': 'none',
})

def clean(ax, grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BLACK)
    ax.spines['bottom'].set_color(BLACK)
    if grid:
        ax.grid(True, color='#E8E8E8', linewidth=0.55, zorder=0)
    ax.tick_params(colors=BLACK)
    ax.xaxis.label.set_color(BLACK); ax.yaxis.label.set_color(BLACK)

def panel(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontweight='bold', fontsize=11,
            va='top', ha='left', color=BLACK)

def save_all(fig, stem, dpi=600, png=True, tif=True, svg=True):
    fig.savefig(OUT/f'{stem}.pdf', bbox_inches='tight')
    if svg:
        fig.savefig(OUT/f'{stem}.svg', bbox_inches='tight')
    if png:
        fig.savefig(OUT/f'{stem}.png', dpi=dpi, bbox_inches='tight')
    if tif:
        fig.savefig(OUT/f'{stem}.tif', dpi=dpi, bbox_inches='tight', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def mol_img(smiles, size=(500,300), legend=''):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(smiles)
    rdDepictor.Compute2DCoords(m)
    opts = Draw.MolDrawOptions()
    opts.clearBackground = False
    opts.padding = 0.06
    opts.bondLineWidth = 1.5
    opts.legendFontSize = 18
    return Draw.MolToImage(m, size=size, options=opts, legend=legend)

def put_mol(ax, smiles, label=None, size=(500,300)):
    im = mol_img(smiles,size=size)
    ax.imshow(im)
    ax.axis('off')
    if label:
        ax.text(0.5, -0.02, label, transform=ax.transAxes, ha='center', va='top', fontsize=7.0, color=BLACK)

# ---------------- Figure 1: workflow ----------------
def figure1():
    fig = plt.figure(figsize=(7.2,7.3))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.45,1.05,1.0,1.18], hspace=0.38)

    # A: chemical-state and metabolic expansion shown as parallel examples
    axA = fig.add_subplot(gs[0]); axA.axis('off'); panel(axA,'A',x=-0.02,y=1.03)
    axA.text(0.03,0.94,'Chemical-state and metabolic expansion',fontweight='bold',fontsize=9.2,va='top')
    # Left: state enumeration example (orphenadrine)
    axA.text(0.25,0.79,'state enumeration',ha='center',fontsize=7.0,color=MID)
    ia=axA.inset_axes([0.04,0.20,0.19,0.50]); put_mol(ia,'Cc1ccccc1C(OCCN(C)C)c1ccccc1','neutral state',size=(420,250))
    ib=axA.inset_axes([0.28,0.20,0.19,0.50]); put_mol(ib,'Cc1ccccc1C(OCC[NH+](C)C)c1ccccc1','protonated state',size=(420,250))
    axA.add_patch(FancyArrowPatch((0.225,0.46),(0.285,0.46),arrowstyle='<|-|>',mutation_scale=10,lw=0.8,color=MID))
    # Right: metabolic transformation example (acetaminophen -> NAPQI)
    axA.text(0.75,0.79,'metabolic transformation',ha='center',fontsize=7.0,color=MID)
    ic=axA.inset_axes([0.53,0.20,0.18,0.50]); put_mol(ic,'CC(=O)Nc1ccc(O)cc1','acetaminophen',size=(400,250))
    idd=axA.inset_axes([0.79,0.20,0.17,0.50]); put_mol(idd,'CC(=O)N=C1C=CC(=O)C=C1','NAPQI',size=(400,250))
    axA.add_patch(FancyArrowPatch((0.705,0.46),(0.79,0.46),arrowstyle='-|>',mutation_scale=11,lw=0.9,color=MID))
    axA.text(0.03,0.05,'Nodes are chemically explicit states; precursor-product edges preserve transformation context.',fontsize=7.0,color=BLACK)

    # B: quantum response
    axB = fig.add_subplot(gs[1]); axB.axis('off'); panel(axB,'B',x=-0.02,y=1.05)
    axB.text(0.03,0.94,'Node quantum chemistry and edge-resolved flux',fontweight='bold',fontsize=9.2,va='top')
    axB.text(0.08,0.55,'GFN2-xTB\n+ ALPB(water)',ha='center',va='center',fontsize=8.0,
             bbox=dict(boxstyle='round,pad=0.35',fc=PALE,ec=LIGHT,lw=0.8))
    axB.add_patch(FancyArrowPatch((0.19,0.55),(0.32,0.55),arrowstyle='-|>',mutation_scale=11,lw=0.9,color=MID))
    items=[('Δω','electrophilicity',CRIMSON),('ΔIP','ionization',TEAL),('ΔEA','electron affinity',PURPLE),('ΔG$_{solv}$ / local','other edge responses',MID)]
    xs=[0.39,0.56,0.73,0.89]
    for x,(sym,desc,c) in zip(xs,items):
        axB.text(x,0.60,sym,ha='center',va='center',fontsize=10,fontweight='bold',color=c)
        axB.text(x,0.36,desc,ha='center',va='center',fontsize=6.3,color=BLACK)
    axB.text(0.03,0.08,'For each edge p → m, transformation features are ΔX = X$_m$ − X$_p$.',fontsize=7.0,color=BLACK)

    # C representations
    axC=fig.add_subplot(gs[2]); axC.axis('off'); panel(axC,'C',x=-0.02,y=1.06)
    axC.text(0.03,0.94,'Locked representation families',fontweight='bold',fontsize=9.2,va='top')
    boxes=[(0.04,0.27,0.20,0.46,'ECFP','2048-bit\nparent structure',BLACK),
           (0.29,0.27,0.20,0.46,'StateCore','8 state descriptors','#777777'),
           (0.54,0.27,0.20,0.46,'FluxCore','16 edge/path descriptors',TEAL),
           (0.79,0.27,0.17,0.46,'StateFluxCore','24 descriptors',PURPLE)]
    for x,y,w,h,title,sub,c in boxes:
        axC.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc='white',ec=LIGHT,lw=0.9))
        axC.text(x+w/2,y+h*0.67,title,ha='center',va='center',fontsize=8,fontweight='bold',color=c)
        axC.text(x+w/2,y+h*0.33,sub,ha='center',va='center',fontsize=6.4,color=BLACK)
    axC.text(0.03,0.08,'The comparison asks whether state/flux chemistry adds information beyond a strong structural baseline.',fontsize=7.0)

    # D validation
    axD=fig.add_subplot(gs[3]); axD.axis('off'); panel(axD,'D',x=-0.02,y=1.04)
    axD.text(0.03,0.94,'Leakage-resistant predictive validation + independent physical validation',fontweight='bold',fontsize=9.2,va='top')
    vals=[('Scaffold CV','DILIrank 2.0\nn = 376\n10 × 5 folds'),('Temporal','strict n = 113\nnovel n = 94'),('DILImap','strict n = 34\nnovel n = 25'),('Cross-endpoint','genotoxicity\nn = 5,367'),('DFT check','24 edges\n144 Q-Chem jobs')]
    x0s=np.linspace(0.04,0.80,5)
    for i,(x,(title,sub)) in enumerate(zip(x0s,vals)):
        w=0.16
        axD.add_patch(FancyBboxPatch((x,0.20),w,0.53,boxstyle='round,pad=0.012',fc='white',ec=LIGHT,lw=0.9))
        axD.text(x+w/2,0.58,title,ha='center',va='center',fontsize=7.4,fontweight='bold',color=BLACK)
        axD.text(x+w/2,0.36,sub,ha='center',va='center',fontsize=6.5,color=BLACK,linespacing=1.3)
        if i<4:
            axD.add_patch(FancyArrowPatch((x+w+0.008,0.47),(x0s[i+1]-0.008,0.47),arrowstyle='-|>',mutation_scale=9,lw=0.7,color=LIGHT))
    axD.text(0.03,0.04,'Predictive value and physical fidelity are evaluated as separate claims.',fontsize=7.0,color=BLACK)
    save_all(fig,'Figure1_StateFlux_Workflow',dpi=600)

# ---------------- Figure 2: DILI validation ----------------
def figure2():
    F=pd.read_csv(DILI_FOLD); E=pd.read_csv(EXTERNAL)
    with open(BOOT) as f: B=json.load(f)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.1),gridspec_kw={'hspace':0.42,'wspace':0.34})
    rng=np.random.default_rng(11)
    for ax,metric,lab,letter in [(axs[0,0],'auroc','AUROC','A'),(axs[0,1],'auprc','AUPRC','B')]:
        data=[]
        for i,m in enumerate(MODELS,1):
            y=F.loc[F.model==m,metric].dropna().to_numpy(); data.append(y)
            x=i+rng.uniform(-0.11,0.11,len(y))
            ax.scatter(x,y,s=8,alpha=.20,color=MODEL_COLORS[i-1],edgecolors='none',zorder=2)
        bp=ax.boxplot(data,positions=np.arange(1,5),widths=.48,patch_artist=True,showfliers=False,
                      medianprops=dict(color=BLACK,lw=1),whiskerprops=dict(color=MID,lw=.75),capprops=dict(color=MID,lw=.75))
        for patch,c in zip(bp['boxes'],MODEL_COLORS): patch.set_facecolor(c); patch.set_alpha(.23); patch.set_edgecolor(c); patch.set_linewidth(.8)
        for i,y in enumerate(data,1): ax.scatter(i,np.mean(y),s=27,marker='D',facecolor='white',edgecolor=MODEL_COLORS[i-1],linewidth=.9,zorder=4)
        ax.set_xticks(range(1,5),MODEL_LABELS,rotation=18,ha='right')
        ax.set_ylabel(lab); clean(ax); panel(ax,letter)
        ax.set_title('50 scaffold-grouped outer-fold estimates',fontweight='normal',pad=5)
        if metric in ('auroc','auprc'): ax.set_ylim(max(.55,min(map(np.min,data))-.05),min(1.0,max(map(np.max,data))+.04))
    # paired delta
    ax=axs[1,0]
    keys=['p_ECFP_StateCore','p_ECFP_FluxCore','p_ECFP_StateFluxCore']; labels=['+ StateCore','+ FluxCore','+ StateFluxCore']; cols=MODEL_COLORS[1:]
    ys=[3,2,1]
    for y,k,l,c in zip(ys,keys,labels,cols):
        rec=B['results'][k]
        point=rec['delta_point']; lo=rec['ci_low']; hi=rec['ci_high']
        ax.plot([lo,hi],[y,y],color=c,lw=1.8,zorder=2)
        ax.scatter(point,y,s=36,color=c,edgecolor='white',lw=.7,zorder=3)
    ax.axvline(0,color=MID,lw=.8,ls='--')
    ax.set_yticks(ys,labels); ax.set_xlabel('Paired ΔAUROC vs ECFP (95% bootstrap CI)')
    ax.set_xlim(-.055,.025); clean(ax); ax.grid(axis='y',visible=False); panel(ax,'C',x=-0.16,y=1.13)
    # external dot plot
    ax=axs[1,1]
    datasets=['temporal_strict','temporal_scaffold_novel','dilimap_strict','dilimap_scaffold_novel']
    dlabs=['Temporal strict\n(n=113; 21+/92−)','Temporal scaffold-novel\n(n=94; 18+/76−)','DILImap strict\n(n=34; 28+/6−)','DILImap scaffold-novel\n(n=25; 20+/5−)']
    ybase=np.arange(4)[::-1]
    offsets=np.linspace(.18,-.18,4)
    for mi,(m,c,lab) in enumerate(zip(MODELS,MODEL_COLORS,MODEL_LABELS)):
        vals=[E[(E.dataset==d)&(E.model==m)].auroc.iloc[0] for d in datasets]
        ax.scatter(vals,ybase+offsets[mi],s=28,color=c,edgecolor='white',lw=.5,label=lab,zorder=3)
    ax.axvline(.5,color=MID,lw=.8,ls='--')
    ax.set_yticks(ybase,dlabs); ax.set_xlabel('AUROC'); ax.set_xlim(.48,1.02); clean(ax); ax.grid(axis='y',visible=False); panel(ax,'D',x=-0.16,y=1.13)
    handles=[Line2D([0],[0],marker='o',linestyle='None',markerfacecolor=c,markeredgecolor='white',label=l) for c,l in zip(MODEL_COLORS,MODEL_LABELS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.75,-0.005),frameon=False,ncol=2,handletextpad=.35,columnspacing=.9)
    save_all(fig,'Figure2_DILI_Validation',dpi=600)

# ---------------- Figure 3 Hagan ----------------
def figure3():
    P=pd.read_csv(HAGAN_PRED); F=pd.read_csv(HAGAN_FOLD); M=pd.read_csv(HAGAN_MET)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),gridspec_kw={'hspace':0.45,'wspace':0.35})
    y=P.y_true.to_numpy(); x=P.p_ECFP.to_numpy(); z=P.p_ECFP_StateFluxCore.to_numpy(); shift=z-x
    # A
    ax=axs[0,0]
    neg=y==0; pos=y==1
    ax.scatter(x[neg],z[neg],s=5,color='#B7B7B7',alpha=.35,edgecolors='none',label='Negative')
    ax.scatter(x[pos],z[pos],s=6,color=CRIMSON,alpha=.35,edgecolors='none',label='Positive')
    ax.plot([0,1],[0,1],'--',color=MID,lw=.8); ax.set(xlim=(0,1),ylim=(0,1),xlabel='ECFP predicted probability',ylabel='StateFluxCore predicted probability')
    clean(ax); panel(ax,'A',x=-0.14,y=1.055); ax.legend(frameon=False,loc='lower right',markerscale=1.8)
    ax.set_title('Independent genotoxicity set (n = 5,367)',fontweight='normal',pad=7)
    # B
    ax=axs[0,1]
    ax.scatter(x[neg],shift[neg],s=5,color='#B7B7B7',alpha=.28,edgecolors='none')
    ax.scatter(x[pos],shift[pos],s=6,color=CRIMSON,alpha=.30,edgecolors='none')
    ax.axhline(0,color=MID,lw=.8); ax.set(xlim=(0,1),xlabel='ECFP predicted probability',ylabel='StateFluxCore − ECFP probability')
    clean(ax); panel(ax,'B',x=-0.14,y=1.055); ax.set_title('Compound-level prediction shift',fontweight='normal',pad=7)
    # C distributions
    ax=axs[1,0]
    bins=np.linspace(np.percentile(shift,.5),np.percentile(shift,99.5),42)
    ax.hist(shift[neg],bins=bins,density=True,histtype='step',lw=1.15,color=MID,label='Negative')
    ax.hist(shift[pos],bins=bins,density=True,histtype='step',lw=1.2,color=CRIMSON,label='Positive')
    ax.axvline(0,color=MID,lw=.75,ls='--'); ax.set_xlabel('StateFluxCore − ECFP probability'); ax.set_ylabel('Density')
    clean(ax); panel(ax,'C',x=-0.14,y=1.055); ax.legend(frameon=False); ax.set_title('Distribution of model-induced score changes',fontweight='normal',pad=6)
    # D metric dot plot
    ax=axs[1,1]
    metrics=['auroc','auprc','mcc','brier']; mlabs=['AUROC ↑','AUPRC ↑','MCC ↑','Brier ↓']; ypos=np.arange(4)[::-1]
    offsets=np.linspace(.15,-.15,4)
    for mi,(m,c,lab) in enumerate(zip(MODELS,MODEL_COLORS,MODEL_LABELS)):
        row=M[M.model==m].iloc[0]
        vals=[row[k] for k in metrics]
        ax.scatter(vals,ypos+offsets[mi],s=28,color=c,edgecolor='white',lw=.5,label=lab,zorder=3)
    ax.set_yticks(ypos,mlabs); ax.set_xlim(.15,.82); ax.set_xlabel('Metric value'); clean(ax); ax.grid(axis='y',visible=False); panel(ax,'D',x=-0.14,y=1.055)
    ax.legend(frameon=False,loc='lower right',ncol=2,handletextpad=.4,columnspacing=.8)
    ax.set_title('Aggregate transfer performance',fontweight='normal',pad=6)
    save_all(fig,'Figure3_Genotoxicity_Transfer',dpi=600)

# ---------------- Figure 4 DFT ----------------
def scatter_dft(ax,df,xcol,ycol,xlab,ylab,letter):
    allx=df[xcol].to_numpy(); ally=df[ycol].to_numpy()
    lo=min(allx.min(),ally.min()); hi=max(allx.max(),ally.max()); pad=.08*max(hi-lo,.3); lim=(lo-pad,hi+pad)
    ax.plot(lim,lim,color=LIGHT,lw=.9,label='identity')
    for cls,lbl,c,mk in zip(CLASS_ORDER,CLASS_LABELS,CLASS_COLORS,CLASS_MARKERS):
        s=df[df.broad_class==cls]
        ax.scatter(s[xcol],s[ycol],s=34,color=c,marker=mk,edgecolor='white',lw=.55,label=lbl,zorder=3)
    p=np.polyfit(allx,ally,1); xx=np.linspace(*lim,100); ax.plot(xx,np.polyval(p,xx),'--',color=MID,lw=.9)
    r=np.corrcoef(allx,ally)[0,1]
    rho=pd.Series(allx).corr(pd.Series(ally),method='spearman')
    mae=np.mean(np.abs(ally-allx))
    txt=f'Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nMAE = {mae:.3f} eV'
    ax.text(.04,.96,txt,transform=ax.transAxes,ha='left',va='top',fontsize=6.7,
            bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.82))
    ax.set(xlim=lim,ylim=lim,xlabel=xlab,ylabel=ylab); clean(ax); panel(ax,letter)
    return lim

def figure4():
    df=pd.read_csv(DFT24)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),gridspec_kw={'hspace':.43,'wspace':.36})
    lim=scatter_dft(axs[0,0],df,'xTB_delta_omega_eV','DFT_delta_omega_eV','Δω$_{xTB}$ (eV)','Δω$_{DFT}$ (eV)','A')
    # label sentinel with conservative offset
    s=df[df.edge=='R05E014'].iloc[0]
    axs[0,0].annotate('Acetaminophen → NAPQI',(s.xTB_delta_omega_eV,s.DFT_delta_omega_eV),xytext=(-82,-20),textcoords='offset points',fontsize=6.5,color=BLACK,
                      arrowprops=dict(arrowstyle='-',lw=.6,color=MID),ha='left',va='top')
    axs[0,0].legend(frameon=False,loc='lower right',handletextpad=.3,borderaxespad=.3)
    scatter_dft(axs[0,1],df,'xTB_delta_IP_eV','DFT_delta_IP_eV','ΔIP$_{xTB}$ (eV)','ΔIP$_{DFT}$ (eV)','B')
    scatter_dft(axs[1,0],df,'xTB_delta_EA_eV','DFT_delta_EA_eV','ΔEA$_{xTB}$ (eV)','ΔEA$_{DFT}$ (eV)','C')
    # Bland-Altman for delta omega
    ax=axs[1,1]
    x=df.xTB_delta_omega_eV.to_numpy(); y=df.DFT_delta_omega_eV.to_numpy(); mean=(x+y)/2; diff=x-y
    bias=diff.mean(); sd=diff.std(ddof=1); lo=bias-1.96*sd; hi=bias+1.96*sd
    for cls,c,mk in zip(CLASS_ORDER,CLASS_COLORS,CLASS_MARKERS):
        s=df[df.broad_class==cls]; xm=(s.xTB_delta_omega_eV+s.DFT_delta_omega_eV)/2; yd=s.xTB_delta_omega_eV-s.DFT_delta_omega_eV
        ax.scatter(xm,yd,s=34,color=c,marker=mk,edgecolor='white',lw=.55,zorder=3)
    ax.axhline(bias,color=BLACK,lw=.9); ax.axhline(lo,color=MID,lw=.8,ls='--'); ax.axhline(hi,color=MID,lw=.8,ls='--')
    ax.text(.04,.96,f'bias = {bias:.3f} eV\n95% limits: {lo:.3f} to {hi:.3f} eV',transform=ax.transAxes,va='top',fontsize=6.7,
            bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.82))
    ax.set_xlabel('Mean Δω by xTB and DFT (eV)'); ax.set_ylabel('xTB − DFT Δω (eV)'); clean(ax); panel(ax,'D')
    save_all(fig,'Figure4_xTB_DFT_Validation',dpi=600)

# ---------------- Figure 5 case studies ----------------
def figure5():
    df=pd.read_csv(DFT24).set_index('edge')
    cases=[('R05E014','Acetaminophen oxidation','Acetaminophen','NAPQI'),('R05E005','Orphenadrine O-dealkylation','Orphenadrine','O-dealkylated product'),('R05E021','Mercaptopurine S-methylation','Mercaptopurine','6-methylmercaptopurine')]
    # Compact three-row layout: preserve molecule readability while eliminating the
    # excessive vertical whitespace that appeared in the manuscript rendering.
    fig=plt.figure(figsize=(7.2,5.8))
    gs=gridspec.GridSpec(3,5,figure=fig,width_ratios=[1.72,.34,1.72,.22,1.34],hspace=.23,wspace=.08)
    fig.subplots_adjust(left=.055,right=.955,top=.91,bottom=.085)
    row_axes=[]
    for ri,(edge,title,plab,qlab) in enumerate(cases):
        row=df.loc[edge]
        axp=fig.add_subplot(gs[ri,0]); put_mol(axp,row.precursor_smiles,plab,size=(500,300))
        axarr=fig.add_subplot(gs[ri,1]); axarr.axis('off'); axarr.add_patch(FancyArrowPatch((.05,.5),(.95,.5),arrowstyle='-|>',mutation_scale=12,lw=.9,color=BLACK)); short_mech=['oxidation','O-dealkylation','S-methylation'][ri]; axarr.text(.5,.69,short_mech,ha='center',va='bottom',fontsize=5.9,color=MID)
        axq=fig.add_subplot(gs[ri,2]); put_mol(axq,row.product_smiles,qlab,size=(500,300))
        axsep=fig.add_subplot(gs[ri,3]); axsep.axis('off')
        axv=fig.add_subplot(gs[ri,4]);
        vals=[row.xTB_delta_omega_eV,row.DFT_delta_omega_eV]
        lo=min(vals)-max(.25,.12*abs(max(vals)-min(vals))); hi=max(vals)+max(.25,.12*abs(max(vals)-min(vals)))
        if abs(hi-lo)<.8: lo-=.3; hi+=.3
        axv.plot(vals,[0,0],color=LIGHT,lw=2,zorder=1)
        axv.scatter(vals[0],0,s=42,color=CRIMSON,edgecolor='white',lw=.6,zorder=3,label='GFN2-xTB')
        axv.scatter(vals[1],0,s=42,facecolor='white',edgecolor=BLACK,lw=1.0,zorder=3,label='ωB97X-D')
        axv.axvline(0,color='#E1E1E1',lw=.7,ls='--'); axv.set_xlim(lo,hi); axv.set_ylim(-.40,.56); axv.set_yticks([]); axv.set_xlabel('Δω (eV)' if ri==2 else '',labelpad=2); axv.tick_params(axis='x',pad=2); clean(axv); axv.grid(axis='y',visible=False)
        axv.text(.03,.92,f'xTB  {vals[0]:+.3f}\nDFT  {vals[1]:+.3f}',transform=axv.transAxes,ha='left',va='top',fontsize=6.6)
        # Row title and letter are aligned consistently above the chemistry rather
        # than consuming a separate band of vertical whitespace.
        axp.text(-.08,1.035,chr(ord('A')+ri),transform=axp.transAxes,fontweight='bold',fontsize=11,va='top')
        axp.text(.02,1.035,title,transform=axp.transAxes,fontweight='bold',fontsize=8.5,va='top')
        row_axes.append(axp)
    # Subtle separators placed exactly between rows.
    fig.canvas.draw()
    for ri in range(2):
        upper_bottom=row_axes[ri].get_position().y0
        lower_top=row_axes[ri+1].get_position().y1
        ysep=(upper_bottom+lower_top)/2
        fig.add_artist(Line2D([.055,.955],[ysep,ysep],transform=fig.transFigure,color='#E4E4E4',lw=.7))
    handles=[Line2D([0],[0],marker='o',linestyle='None',markerfacecolor=CRIMSON,markeredgecolor='white',label='GFN2-xTB'),
             Line2D([0],[0],marker='o',linestyle='None',markerfacecolor='white',markeredgecolor=BLACK,label='ωB97X-D')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.955,.988),frameon=False,ncol=2,columnspacing=.9,handletextpad=.3)
    fig.text(.055,.018,'Representative transformations span strong electrophilic activation, moderate activation, and slight deactivation while preserving the direction of the higher-level DFT response.',fontsize=7.0,color=BLACK)
    save_all(fig,'Figure5_Mechanistic_Case_Studies',dpi=600)

# ---------------- TOC ----------------
def toc():
    # ACS TOC maximum: 3.25 x 1.75 in. Keep all content well within the canvas.
    fig=plt.figure(figsize=(3.25,1.75))
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ia=ax.inset_axes([.025,.39,.17,.40]); put_mol(ia,'CC(=O)Nc1ccc(O)cc1',None,size=(240,160))
    ib=ax.inset_axes([.255,.39,.17,.40]); put_mol(ib,'CC(=O)N=C1C=CC(=O)C=C1',None,size=(240,160))
    ax.add_patch(FancyArrowPatch((.195,.59),(.255,.59),arrowstyle='-|>',mutation_scale=8,lw=.8,color=BLACK))
    ax.text(.225,.76,'metabolism',ha='center',fontsize=5.2,color=MID)
    ax.text(.49,.66,'Δω',fontsize=10,fontweight='bold',ha='center',color=CRIMSON)
    ax.text(.49,.49,'xTB ≈ DFT',fontsize=6.6,fontweight='bold',ha='center',color=BLACK)
    ax.text(.49,.33,'quantum-chemical fidelity',fontsize=5.0,ha='center',color=MID)
    ax.text(.78,.80,'toxicity prediction',fontsize=5.0,ha='center',color=MID)
    ax.add_patch(Rectangle((.68,.48),.05,.20,fc=BLACK,ec='none'))
    ax.add_patch(Rectangle((.82,.48),.05,.18,fc=PURPLE,ec='none'))
    ax.text(.705,.41,'ECFP',fontsize=5.2,ha='center')
    ax.text(.845,.41,'StateFlux',fontsize=5.2,ha='center')
    ax.text(.78,.26,'limited incremental value',fontsize=5.0,ha='center',fontweight='bold',color=PURPLE)
    ax.text(.50,.08,'Quantum-chemical fidelity ≠ incremental predictive value',fontsize=5.6,ha='center',fontweight='bold',color=BLACK)
    for ext in ['png','tif','pdf','svg']:
        path=OUT/f'TOC_StateFlux.{ext}'
        if ext=='png': fig.savefig(path,dpi=600,bbox_inches=None,pad_inches=0)
        elif ext=='tif': fig.savefig(path,dpi=600,bbox_inches=None,pad_inches=0,pil_kwargs={'compression':'tiff_lzw'})
        else: fig.savefig(path,bbox_inches=None,pad_inches=0)
    plt.close(fig)

if __name__=='__main__':
    figure1(); figure2(); figure3(); figure4(); figure5(); toc()
    print('Generated figures in',OUT)
