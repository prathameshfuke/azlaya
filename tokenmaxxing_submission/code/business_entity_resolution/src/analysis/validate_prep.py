import pandas as pd, numpy as np, csv, os
from rapidfuzz import fuzz
P='/home/user/amazon_ml/student_resource/dataset_processed'
rd=lambda p: pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE,escapechar='\\')
for sp in ['train','test']:
    for s in (1,2,3):
        d=rd(f'{P}/{sp}/{sp}_source{s}.tsv'); print(sp,s,d.shape, 'empty embed:',(d.embed_text.str.strip()=='').mean().round(5), 'house filled:',(d.house_no!='').mean().round(3))
        if sp=='test' and s==1: t1=d
fr=t1[t1.country=='France']; print('France S1 parse: house',(fr.house_no!='').mean().round(3),' legal',(fr.legal_form!='').mean().round(3))
s1=rd(f'{P}/train/train_source1.tsv').set_index('entity_id')
o=pd.concat([rd(f'{P}/train/train_source{s}.tsv') for s in (2,3)]).set_index('entity_id')
gt=rd(f'{P}/train/train_ground_truth.tsv'); p=gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m'); p=p[p.m!='']
owner=dict(zip(p.m,p.source1_entity_id))
ps=p.sample(300000,random_state=0); A=s1.loc[ps.source1_entity_id]; B=o.loc[ps.m]
def rates(A,B,lab):
    h=(A.house_no.values==B.house_no.values)&(A.house_no.values!=''); hb=(A.house_base.values==B.house_base.values)&(A.house_base.values!='')
    l=A.legal_form.values==B.legal_form.values
    nc=A.name_core.values==B.name_core.values
    es=np.array([fuzz.token_set_ratio(a,b) for a,b in zip(A.embed_text.values[:50000],B.embed_text.values[:50000])])
    print(f'{lab:28s} house_eq={h.mean():.3f} house_base_eq={hb.mean():.3f} legal_eq={l.mean():.3f} name_core_eq={nc.mean():.3f} embed_tsr_median={np.median(es):.1f}')
for c in ['US','India']:
    m=(A.country==c).values; rates(A[m],B[m],f'TRUE pairs {c}')
    m2=m&(B.is_indic=='1').values
    if m2.sum(): rates(A[m2],B[m2],f'TRUE pairs {c} Indic names')
# raw baseline for name equality
# hard negatives: same name_core, US, not owner, similar address
ou=o[o.country=='US'].sample(400000,random_state=0).reset_index()
su=s1[s1.country=='US'].reset_index()[['entity_id','name_core','house_no','house_base','legal_form','addr_clean']].rename(columns={'entity_id':'s1id'})
m=ou.merge(su,on='name_core',suffixes=('','_1'))
m=m[m.entity_id.map(owner)!=m.s1id]
m['atsr']=[fuzz.token_set_ratio(a,b) for a,b in zip(m.addr_clean,m.addr_clean_1)]
m=m[m.atsr>=70]
h=(m.house_no==m.house_no_1)&(m.house_no!=''); hb=(m.house_base==m.house_base_1)&(m.house_base!=''); l=m.legal_form==m.legal_form_1
print(f'HARD NEGATIVES US (n={len(m):,})   house_eq={h.mean():.3f} house_base_eq={hb.mean():.3f} legal_eq={l.mean():.3f}')
