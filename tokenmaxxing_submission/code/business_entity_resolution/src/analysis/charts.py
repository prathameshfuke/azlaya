import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B,O,A='#2a78d6','#eb6834','#1baf7a'; T1,T2,GR='#0b0b0b','#52514e','#e6e5e1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':GR,'axes.labelcolor':T2,'xtick.color':T2,'ytick.color':T2,
  'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GR,'grid.linewidth':0.6,'axes.axisbelow':True,'figure.dpi':200,'savefig.bbox':'tight'})
def fin(ax,title,fn,ylab=None):
    ax.set_title(title,loc='left',color=T1,fontsize=10,fontweight='bold'); 
    if ylab: ax.set_ylabel(ylab)
    ax.figure.savefig('fig/'+fn); plt.close(ax.figure)
def bars(ax,x,y,c,w=0.8,**k):
    return ax.bar(x,y,width=w,color=c,edgecolor='white',linewidth=1.5,**k)
# 1 matches per S1
d={0:123247,1:119157,2:375212,3:530841,4:484115,5:321957,6:164868,7:63968,8:18680,9:4205,10:534,11:37}
fig,ax=plt.subplots(figsize=(6.4,2.6)); x=list(d); y=np.array(list(d.values()))/2206821*100
bars(ax,x,y,B); ax.set_xticks(x); ax.set_xlabel('Number of S2+S3 matches per S1 entity'); ax.grid(axis='x',visible=False)
for xi,yi in zip(x,y):
    if xi in (0,3): ax.text(xi,yi+0.5,f'{yi:.1f}%',ha='center',color=T1,fontsize=8)
ax.text(0,y[0]+3,'singletons',ha='center',color=T2,fontsize=7.5)
fin(ax,'Matches per S1 entity (train, share of 2.21M entities)','matches.png','% of S1 entities')
# 2 country mix
cm={'Train S1':(1323633,883188,0),'Train S2':(3016817,2017799,0),'Train S3':(3170056,2115547,0),
    'Test S1':(663106,809986,259452),'Test S2':(1871330,2312565,703378),'Test S3':(1945701,2405000,731615)}
fig,ax=plt.subplots(figsize=(6.4,2.6)); labs=list(cm); v=np.array([cm[k] for k in labs],float); v=v/v.sum(1,keepdims=True)*100
left=np.zeros(len(labs))
for i,(n,c) in enumerate(zip(['US','India','France'],[B,O,A])):
    ax.barh(labs,v[:,i],left=left,color=c,edgecolor='white',linewidth=2,label=n,height=0.65)
    for j in range(len(labs)):
        if v[j,i]>6: ax.text(left[j]+v[j,i]/2,j,f'{v[j,i]:.0f}%',ha='center',va='center',color='white',fontsize=7.5,fontweight='bold')
    left+=v[:,i]
ax.invert_yaxis(); ax.set_xlim(0,100); ax.grid(axis='y',visible=False); ax.set_xlabel('% of records')
ax.legend(ncol=3,frameon=False,loc='lower left',bbox_to_anchor=(0,1.0),fontsize=8)
ax.set_title('Country mix by file',loc='left',color=T1,fontsize=10,fontweight='bold',pad=18); fig.savefig('fig/country.png'); plt.close(fig)
# 3 name similarity true matches
f=pd.read_parquet('pairfeat.parquet')
fig,ax=plt.subplots(figsize=(6.4,2.6)); bins=np.arange(0,105,5)
for c,col in [('US',B),('India',O)]:
    h,_=np.histogram(f[f.c==c].name_tsr,bins=bins); ax.plot(bins[:-1]+2.5,h/h.sum()*100,color=col,lw=2,marker='o',ms=4,label=c)
ax.set_yscale('log'); ax.set_xlabel('Name token-set similarity (0-100) after normalisation'); ax.legend(frameon=False)
fin(ax,'Name similarity of TRUE matched pairs (200k sample)','namesim.png','% of pairs (log scale)')
# 4 hard-negative house diff fingerprint
neg={0:353,1:2905,2:2822,3:2846,4:2836,5:2689,6:21,7:2806,8:18,9:2800,10:15,11:2880,12:20}
pos={1:507,2:574,3:42,4:59,5:46,6:39,7:35,8:54,9:44,10:52,11:53,12:43}
fig,axs=plt.subplots(1,2,figsize=(6.6,2.5),sharey=False)
x=list(neg); bars(axs[0],x,list(neg.values()),O); axs[0].set_xticks(x); axs[0].set_title('Hard negatives',loc='left',fontsize=9,color=T1)
axs[0].set_xlabel('|house no. difference|'); axs[0].set_ylabel('pairs'); axs[0].grid(axis='x',visible=False)
x2=list(pos); bars(axs[1],x2,list(pos.values()),B); axs[1].set_xticks(range(0,13)); axs[1].set_title('True matches (diff > 0 only; 87.7% have diff = 0)',loc='left',fontsize=9,color=T1)
axs[1].set_xlabel('|house no. difference|'); axs[1].grid(axis='x',visible=False)
fig.suptitle('House-number difference: the distractor fingerprint (US, same-name & similar-address pairs)',x=0.02,ha='left',fontsize=10,fontweight='bold',color=T1,y=1.04)
fig.tight_layout(); fig.savefig('fig/housediff.png'); plt.close(fig)
# 5 pos vs neg signals
fig,ax=plt.subplots(figsize=(6.4,2.5)); cats=['House no. equal','House no. differs by 1-10','Legal suffix identical']
p=[87.7,1.4,65.1]; n=[1.1,62.4,1.5]; xx=np.arange(3)
bars(ax,xx-0.2,p,B,w=0.38,label='True matches'); bars(ax,xx+0.2,n,O,w=0.38,label='Hard negatives')
for i in range(3): ax.text(xx[i]-0.2,p[i]+2,f'{p[i]}%',ha='center',fontsize=8,color=T1); ax.text(xx[i]+0.2,n[i]+2,f'{n[i]}%',ha='center',fontsize=8,color=T1)
ax.set_xticks(xx,cats); ax.set_ylim(0,115); ax.set_yticks(range(0,101,20)); ax.grid(axis='x',visible=False); ax.legend(frameon=False,ncol=2,loc='upper center')
fin(ax,'Separating true matches from hard negatives','signals.png','% of pairs')
# 6 blocking recall
k=['Normalised name (exact)','House no. + 1st street word','House no. + 2 street words','1st name token + house no.','Union of all four']
r=[50.5,48.7,38.9,52.6,79.8]
fig,ax=plt.subplots(figsize=(6.4,2.4)); ax.barh(k,r,color=[B]*4+[A],edgecolor='white',height=0.6); ax.invert_yaxis(); ax.set_xlim(0,100)
for i,v in enumerate(r): ax.text(v+1,i,f'{v}%',va='center',fontsize=8,color=T1)
ax.axvline(97,color=T2,lw=1); ax.text(96,-0.55,'target ≥97%',ha='right',fontsize=7.5,color=T2)
ax.grid(axis='y',visible=False); ax.set_xlabel('Pair recall on 7.64M true train pairs')
fin(ax,'Recall of simple exact blocking keys','blocking.png')
print('ok')
