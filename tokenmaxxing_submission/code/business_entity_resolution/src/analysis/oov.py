import pandas as pd, numpy as np, re, unicodedata, collections, json
BLOCKS=[('Devanagari',0x900,0x97f),('Bengali',0x980,0x9ff),('Gurmukhi',0xa00,0xa7f),('Gujarati',0xa80,0xaff),('Odia',0xb00,0xb7f),
        ('Tamil',0xb80,0xbff),('Telugu',0xc00,0xc7f),('Kannada',0xc80,0xcff),('Malayalam',0xd00,0xd7f)]
def script(tok):
    for n,a,b in BLOCKS:
        if any(a<=ord(c)<=b for c in tok): return n
    if any(ord(c)>127 and c.isalpha() for c in tok): return 'Latin-accented'
    return 'ASCII'
def toks(s): return re.findall(r'\w+',s.lower())
R={}
def vocab(df,col):
    c=collections.Counter()
    for s in df[col]: c.update(toks(s))
    return c
tr=pd.concat([pd.read_parquet('train_s2.parquet'),pd.read_parquet('train_s3.parquet')])
te=pd.concat([pd.read_parquet('test_s2.parquet'),pd.read_parquet('test_s3.parquet')])
for col in ['business_name','business_address']:
    vtr=vocab(tr,col); vte=vocab(te,col)
    by=collections.defaultdict(lambda:[0,0,0,0])  # test tok occurrences, unseen occ, test types, unseen types
    for t,n in vte.items():
        s=script(t); u=t not in vtr
        by[s][0]+=n; by[s][1]+=n*u; by[s][2]+=1; by[s][3]+=u
    tr_by=collections.Counter()
    for t,n in vtr.items(): tr_by[script(t)]+=n
    print(f'\n== {col}: script | train occ | test occ | test occ unseen in train | test types | unseen types')
    out={}
    for s,(o,uo,ty,ut) in sorted(by.items(),key=lambda x:-x[1][0]):
        print(f'{s:15s} {tr_by[s]:>11,} {o:>11,} {uo/o:7.1%} {ty:>9,} {ut/ty:7.1%}')
        out[s]=dict(train_occ=tr_by[s],test_occ=o,unseen_occ=uo/o,test_types=ty,unseen_types=ut/ty)
    R[col]=out
# country-wise: France vocab vs train
fr=te[te.country=='France']; vfr=vocab(fr,'business_name'); vtr=vocab(tr,'business_name')
tot=sum(vfr.values()); un=sum(n for t,n in vfr.items() if t not in vtr)
print('\nFrance name token occurrences unseen in train:',un/tot)
R['france_name_unseen_occ']=un/tot
print('top unseen France name tokens:',[t for t,n in vfr.most_common(400) if t not in vtr][:40])
va=vocab(fr,'business_address'); vta=vocab(tr,'business_address'); tot=sum(va.values()); un=sum(n for t,n in va.items() if t not in vta)
print('France addr token occ unseen:',un/tot, [t for t,n in va.most_common(300) if t not in vta][:40])
R['france_addr_unseen_occ']=un/tot
# India only, name, by source Indic: row-level share with any unseen indic token
ind=te[te.country=='India']
vtr_i=vocab(tr[tr.country=='India'],'business_name')
def row_unseen(s): 
    ts=[t for t in toks(s) if script(t) not in ('ASCII','Latin-accented')]
    return None if not ts else any(t not in vtr_i for t in ts)
ru=ind.business_name.map(row_unseen).dropna()
print('\nIndia test names with Indic script:',len(ru),' share having >=1 unseen Indic token:',ru.mean())
R['india_indic_rows_with_unseen']=float(ru.mean())
# example unseen Indic tokens (count only, script)
unseen=[(t,n) for t,n in vocab(ind,'business_name').most_common() if t not in vtr_i and script(t) not in ('ASCII','Latin-accented')]
print('top unseen indic tokens freq:',[n for t,n in unseen[:15]])
# Latin (English) India name tokens unseen
lat=[(t,n) for t,n in vocab(ind,'business_name').items() if script(t)=='ASCII']
tot=sum(n for t,n in lat); un=sum(n for t,n in lat if t not in vtr_i); print('India ASCII name tok occ unseen:',un/tot)
R['india_ascii_name_unseen_occ']=un/tot
# US
us=te[te.country=='US']; vtu=vocab(tr[tr.country=='US'],'business_name'); vu=vocab(us,'business_name'); tot=sum(vu.values()); un=sum(n for t,n in vu.items() if t not in vtu)
print('US name tok occ unseen:',un/tot); R['us_name_unseen_occ']=un/tot
json.dump(R,open('oov.json','w'),indent=1)
