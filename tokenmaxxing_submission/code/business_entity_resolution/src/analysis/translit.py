import pandas as pd, re, unicodedata, json, numpy as np
from rapidfuzz import fuzz
from indic_transliteration import sanscript
from indic_transliteration.detect import detect
f=pd.read_parquet('pairfeat.parquet'); f=f[f.indic].copy(); print("indic pairs",len(f))
SCR={'Devanagari':sanscript.DEVANAGARI,'Bengali':sanscript.BENGALI,'Gurmukhi':sanscript.GURMUKHI,'Gujarati':sanscript.GUJARATI,'Oriya':sanscript.ORIYA,
     'Tamil':sanscript.TAMIL,'Telugu':sanscript.TELUGU,'Kannada':sanscript.KANNADA,'Malayalam':sanscript.MALAYALAM}
BLK=[(0x900,sanscript.DEVANAGARI),(0x980,sanscript.BENGALI),(0xa00,sanscript.GURMUKHI),(0xa80,sanscript.GUJARATI),(0xb00,sanscript.ORIYA),
     (0xb80,sanscript.TAMIL),(0xc00,sanscript.TELUGU),(0xc80,sanscript.KANNADA),(0xd00,sanscript.MALAYALAM)]
def scr_of(tok):
    for c in tok:
        o=ord(c)
        for a,s in BLK:
            if a<=o<a+0x80: return s
    return None
def strip(s): return ''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
def simplify(s):
    s=strip(s.lower()); s=re.sub(r'(.)\1+',r'\1',s)          # collapse doubled letters (aa->a)
    s=s.replace('w','v').replace('sh','s').replace('ph','f').replace('z','j')
    return re.sub(r'[^a-z0-9 ]',' ',s)
def translit(name, scheme):
    out=[]
    for t in name.split():
        s=scr_of(t); out.append(sanscript.transliterate(t,s,scheme) if s else t)
    return ' '.join(out)
def norm(s): 
    s=strip(s.lower()); s=re.sub(r'[^a-z0-9 ]',' ',s); return ' '.join(s.split())
res={}
for lab,fn in [('raw (no transliteration)',lambda n:norm(n)),
               ('IAST rule-based',lambda n:norm(translit(n,sanscript.IAST))),
               ('IAST + phonetic simplify',lambda n:simplify(translit(n,sanscript.IAST))),
               ('ITRANS + phonetic simplify',lambda n:simplify(translit(n,sanscript.ITRANS)))]:
    t=f.n2.map(fn); ref=f.n1.map(simplify if 'simplify' in lab else norm)
    sc=np.array([fuzz.token_set_ratio(a,b) for a,b in zip(ref,t)]); ch=np.array([fuzz.ratio(a,b) for a,b in zip(ref,t)])
    res[lab]={'tsr_median':float(np.median(sc)),'tsr_ge80':float((sc>=80).mean()),'char_ratio_median':float(np.median(ch))}
    print(f'{lab:30s} token-set median {np.median(sc):5.1f}  >=80: {(sc>=80).mean():.3f}  char-ratio median {np.median(ch):5.1f}')
ex=f.head(8)
for a,b in zip(ex.n1,ex.n2): print('  S1:',a,' | IAST+simplify:',simplify(translit(b,sanscript.IAST)))
json.dump(res,open('translit.json','w'),indent=1)

def fix(s):
    s=simplify(s)
    s=re.sub(r'm(?=[kgcjtdpbfsvhlr])','n',s)     # anusvara -> n before consonant
    s=re.sub(r'(?<=[a-z]{3})a\b','',s)            # schwa deletion at word end
    return s
def skel(s): return ' '.join(re.sub(r'(?<=.)[aeiou]','',w) for w in s.split())
for lab,fn,rf in [('IAST + simplify + schwa/anusvara',lambda n:fix(translit(n,sanscript.IAST)),fix),
                  ('... + consonant skeleton',lambda n:skel(fix(translit(n,sanscript.IAST))),lambda n:skel(fix(n)))]:
    t=f.n2.map(fn); ref=f.n1.map(rf)
    sc=np.array([fuzz.token_set_ratio(a,b) for a,b in zip(ref,t)]); ch=np.array([fuzz.ratio(a,b) for a,b in zip(ref,t)])
    res[lab]={'tsr_median':float(np.median(sc)),'tsr_ge80':float((sc>=80).mean()),'char_ratio_median':float(np.median(ch))}
    print(f'{lab:34s} token-set median {np.median(sc):5.1f}  >=80: {(sc>=80).mean():.3f}  char-ratio median {np.median(ch):5.1f}')
for a,b in zip(ex.n1,ex.n2): print('  S1:',fix(a),' | fixed:',fix(translit(b,sanscript.IAST)))
# baseline: same metric on non-indic India pairs for reference
g=pd.read_parquet('pairfeat.parquet'); g=g[(~g.indic)&(g.c=='India')]
sc=np.array([fuzz.ratio(fix(a),fix(b)) for a,b in zip(g.n1,g.n2)]); print('reference: Latin India pairs char-ratio median',np.median(sc))
res['latin_india_char_ratio_median']=float(np.median(sc))
json.dump(res,open('translit.json','w'),indent=1)
