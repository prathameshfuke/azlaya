import pandas as pd, re, collections, json
BLOCKS=[('Devanagari',0x900,0x97f),('Bengali',0x980,0x9ff),('Gurmukhi',0xa00,0xa7f),('Gujarati',0xa80,0xaff),('Odia',0xb00,0xb7f),
        ('Tamil',0xb80,0xbff),('Telugu',0xc00,0xc7f),('Kannada',0xc80,0xcff),('Malayalam',0xd00,0xd7f)]
def script(tok):
    for n,a,b in BLOCKS:
        if any(a<=ord(c)<=b for c in tok): return n
    return None
def wtoks(s): return [t for t in re.split(r'[\s,.;:()\[\]|/\-]+',s) if t]
tr=pd.concat([pd.read_parquet('train_s2.parquet'),pd.read_parquet('train_s3.parquet')]); tr=tr[tr.country=='India']
te=pd.concat([pd.read_parquet('test_s2.parquet'),pd.read_parquet('test_s3.parquet')]); te=te[te.country=='India']
R={}
for col in ['business_name','business_address']:
    vtr=collections.Counter(); vte=collections.Counter()
    for s in tr[col]: vtr.update(t for t in wtoks(s) if script(t))
    for s in te[col]: vte.update(t for t in wtoks(s) if script(t))
    by=collections.defaultdict(lambda:[0,0,0,0,0]); trby=collections.Counter(); trty=collections.Counter()
    for t,n in vtr.items(): trby[script(t)]+=n; trty[script(t)]+=1
    for t,n in vte.items():
        s=script(t);u=t not in vtr; by[s][0]+=n;by[s][1]+=n*u;by[s][2]+=1;by[s][3]+=u
    print(f'\n== India {col}: script | train types | test occ | unseen occ | test types | unseen types')
    R[col]={}
    for s,(o,uo,ty,ut,_) in sorted(by.items(),key=lambda x:-x[1][0]):
        print(f'{s:11s} {trty[s]:>8,} {o:>10,} {uo/o:7.2%} {ty:>8,} {ut/ty:7.1%}')
        R[col][s]=dict(train_types=trty[s],test_occ=o,unseen_occ=uo/o,test_types=ty,unseen_types=ut/ty)
    if col=='business_name':
        def rowu(s):
            ts=[t for t in wtoks(s) if script(t)]
            return None if not ts else any(t not in vtr for t in ts)
        ru=te[col].map(rowu).dropna(); R['rows_with_indic']=len(ru); R['rows_with_unseen_indic']=float(ru.mean())
        print('test India names with Indic tokens:',len(ru),' with >=1 unseen Indic word:',ru.mean())
        # fully indic vs code-mixed
        def kind(s):
            ts=wtoks(s); k=[bool(script(t)) for t in ts]
            return None if not any(k) else ('full' if all(k) else 'mixed')
        kk=te[col].map(kind).dropna().value_counts(normalize=True).to_dict(); print('full vs code-mixed:',kk); R['full_vs_mixed']=kk
        # how many distinct scripts per test name row
        top=[n for t,n in vtr.most_common(20)]; print('train top-20 indic word freqs:',top)
        cov=sum(n for t,n in vtr.most_common(2000))/sum(vtr.values()); print('train: top-2000 indic words cover',cov); R['top2000_cover']=cov
json.dump(R,open('oov2.json','w'),indent=1)
