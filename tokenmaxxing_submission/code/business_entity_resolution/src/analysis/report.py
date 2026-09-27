from reportlab.lib.pagesizes import A4
from reportlab.platypus import *
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
F='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',F+'DejaVuSans.ttf')); pdfmetrics.registerFont(TTFont('DVB',F+'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DVI',F+'DejaVuSans-Oblique.ttf')); pdfmetrics.registerFont(TTFont('DVM',F+'DejaVuSansMono.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('DV',normal='DV',bold='DVB',italic='DVI',boldItalic='DVB')
INK=colors.HexColor('#0b0b0b'); SUB=colors.HexColor('#52514e'); ACC=colors.HexColor('#2a78d6'); LINE=colors.HexColor('#d9d8d3'); BG=colors.HexColor('#f4f3ef')
ss=getSampleStyleSheet()
body=ParagraphStyle('b',fontName='DV',fontSize=9.2,leading=13.2,textColor=INK,spaceAfter=5)
bul=ParagraphStyle('bu',parent=body,leftIndent=12,bulletIndent=2,spaceAfter=2.5)
h1=ParagraphStyle('h1',fontName='DVB',fontSize=14,leading=18,textColor=INK,spaceBefore=10,spaceAfter=6,keepWithNext=1)
h2=ParagraphStyle('h2',fontName='DVB',fontSize=10.5,leading=14,textColor=ACC,spaceBefore=8,spaceAfter=4,keepWithNext=1)
cap=ParagraphStyle('c',parent=body,fontSize=8,leading=11,textColor=SUB,spaceAfter=10)
cell=ParagraphStyle('cell',fontName='DV',fontSize=7.8,leading=10,textColor=INK)
cellb=ParagraphStyle('cellb',parent=cell,fontName='DVB')
title=ParagraphStyle('t',fontName='DVB',fontSize=22,leading=27,textColor=INK)
S=[]
P=lambda t,s=body:S.append(Paragraph(t,s))
def B(items):
    for t in items: S.append(Paragraph(t,bul,bulletText='•'))
def T(rows,widths,head=True):
    data=[[Paragraph(str(c),cellb if (head and i==0) else cell) for c in r] for i,r in enumerate(rows)]
    t=Table(data,colWidths=[w*mm for w in widths],repeatRows=1 if head else 0)
    st=[('LINEBELOW',(0,0),(-1,-1),0.4,LINE),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
    if head: st+= [('BACKGROUND',(0,0),(-1,0),BG),('LINEBELOW',(0,0),(-1,0),0.8,SUB)]
    t.setStyle(TableStyle(st)); S.append(t); S.append(Spacer(1,8))
def IMG(fn,w=170,c=None):
    from reportlab.lib.utils import ImageReader
    iw,ih=ImageReader('fig/'+fn).getSize(); S.append(Image('fig/'+fn,width=w*mm,height=w*mm*ih/iw))
    if c: P(c,cap)
    else: S.append(Spacer(1,8))
def KEY(t):
    k=Table([[Paragraph(t,ParagraphStyle('k',parent=body,spaceAfter=0))]],colWidths=[170*mm])
    k.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),BG),('LINEBEFORE',(0,0),(0,-1),2.5,ACC),('LEFTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    S.append(k); S.append(Spacer(1,8))

# ---------- Cover / summary
S.append(Spacer(1,6)); P('Exploratory Data Analysis',title)
P('Amazon ML Challenge 2026 — Business Entity Resolution',ParagraphStyle('st',parent=body,fontSize=12,leading=16,textColor=SUB))
P('Scope: all six source files and the training ground truth (~28M records). Every figure in this report was computed from the provided data.',cap)
S.append(HRFlowable(width='100%',color=LINE,thickness=0.8,spaceAfter=8))
P('Key findings',h1)
B(['<b>Heavily linked data.</b> Only 5.6% of Source 1 entities are singletons; the average entity has 3.46 matches (up to 11). Recall matters as well as precision.',
   '<b>Every S2/S3 record belongs to at most one S1 entity</b> (7.64M matched IDs, all unique). About 26% of S2/S3 records match nothing and act as distractors.',
   '<b>Country agrees in 100% of true pairs</b>, so it is a free blocking key. The test set adds France (15% of test S1), which has no labels in training.',
   '<b>Name alone is unreliable.</b> 50% of S1 entities share a normalised name with another S1 entity, and about 6% of true matches carry a completely unrelated name (a brand, an acronym or a domain).',
   '<b>Distractors are adversarial near-copies</b>: same name and street, a house number off by exactly 1–5, 7, 9 or 11, and a changed legal suffix. House-number equality (87.7% of true pairs vs 1.1% of hard negatives) and legal-suffix equality (65% vs 1.5%) are the strongest precision signals.',
   '<b>Simple exact blocking keys reach only 79.8% pair recall</b> (an upper-bound macro F0.5 of about 0.91). Fuzzy or embedding-based candidate generation is required.',
   '<b>India records in S2/S3 are 9–11% non-Latin script</b> (Devanagari, Telugu, Tamil, Kannada, Bengali): transliterations of the English name. Transliteration is a necessary preprocessing step.',
   '<b>No leakage</b>: entity IDs and row order are uncorrelated with matches, and train and test share no IDs.'])

# ---------- 1 Problem
P('1. Problem and evaluation',h1)
P('Source 1 is a deduplicated reference list of businesses. For every S1 entity we must return all S2 and S3 records that refer to the same real-world business, which may be none, one or many. The three sources share no identifiers; each record has only <i>business_name</i>, <i>business_address</i> and <i>country</i>.')
P('The metric is <b>F0.5, computed per S1 entity and then macro-averaged</b> over all entities, singletons included. F0.5 = 1.25·P·R / (0.25·P + R) weights precision twice as heavily as recall. A singleton scores 1.0 for an empty prediction and 0.0 for any prediction. Two files are submitted: <font name="DVM">matching_results.tsv</font> (scored) and <font name="DVM">candidate_pairs.tsv</font> (used to audit blocking recall and reduction ratio). Constraints: MIT/Apache-2.0 models of at most 8B parameters, and no external data lookup (including geocoding).')
KEY('<b>Implication.</b> Because singletons are only 5.6% and the average entity has 3–4 matches, predicting only the most confident match leaves a lot of F0.5 on the table. With perfect precision and 50% recall an entity scores 0.83; with 100% recall it scores 1.0.')

# ---------- 2 Overview
P('2. Dataset overview',h1)
T([['File','Rows','US','India','France','Name mean len','Addr mean len'],
   ['train_source1','2,206,821','1,323,633','883,188','—','24.0','52.1'],
   ['train_source2','5,034,616','3,016,817','2,017,799','—','25.1','46.2'],
   ['train_source3','5,285,603','3,170,056','2,115,547','—','25.2','46.7'],
   ['test_source1','1,732,544','663,106','809,986','259,452','23.8','57.2'],
   ['test_source2','4,887,273','1,871,330','2,312,565','703,378','25.7','50.4'],
   ['test_source3','5,082,316','1,945,701','2,405,000','731,615','25.7','48.7']],[30,22,22,22,20,27,27])
IMG('country.png',160,'Figure 1. The country mix shifts between train and test: India becomes the largest group and France (unseen in training) makes up 14–15%. S2 and S3 are each about 2.3–2.9× the size of S1.')
P('File integrity: row counts equal the line counts exactly, so no field contains embedded tabs or newlines. Read every column as a string with <font name="DVM">keep_default_na=False</font>; otherwise names such as "NULL" or "NA" are silently turned into missing values.')

# ---------- 3 GT
P('3. Ground-truth structure',h1)
IMG('matches.png',160,'Figure 2. Distribution of matches per S1 entity. The mode is 3; only 5.6% are singletons.')
T([['Statistic','Value'],['Singleton S1 entities','123,247 (5.58%); the same rate in US and India'],
   ['Mean matches per S1','3.46 (India 3.46, US 3.46)'],['Max matches from S2 / S3 for one entity','5 / 6, so S2 and S3 are not deduplicated'],
   ['Total true pairs','7,638,365'],['S2/S3 IDs matched to more than one S1','0 (one-to-many only)'],
   ['Share of S2 / S3 records matched to some S1','73.4% / 74.6%'],['Country agreement in true pairs','100%'],
   ['Correlation of numeric ID (or row position) between pair members','0.0001 (no leakage)']],[85,85])

# ---------- 4 Quality
P('4. Field quality and script mix',h1)
T([['File','Empty address','Literal NULL / None in address','Non-ASCII name','Devanagari name','Other Indic name','US address all-caps'],
   ['train S1','0.0%','0.0%','0.0%','0.0%','0.0%','0.0%'],
   ['train S2','3.4%','2.6%','15.2%','5.4%','4.1%','93.5%'],
   ['train S3','3.3%','2.5%','11.5%','3.0%','2.3%','3.5%'],
   ['test S1','0.0%','0.0%','2.4%*','0.0%','0.0%','0.0%'],
   ['test S2','2.7%','2.2%','19.0%','6.3%','4.9%','93.4%'],
   ['test S3','2.7%','2.1%','14.5%','3.6%','2.8%','2.8%']],[20,22,30,22,24,24,28])
P('* The non-ASCII names in test S1 are French accented names. S1 is otherwise clean, title-cased Latin text. S2 US addresses are almost entirely upper-case; S3 uses full state names ("Maryland") where S1 uses codes ("MD").',cap)
B(['<b>Postal codes are nearly absent</b>: no 6-digit Indian PIN codes appear anywhere, and only about 10% of US addresses carry a 5-digit ZIP, so postal-code blocking is not viable.',
   '<b>India addresses are long and unstructured</b> (4.7 commas on average vs 2.2 for the US), and 7–11% contain landmark phrases (Near, Opp, Behind, Next to).',
   '<b>Empty addresses concentrate in true matches</b> (4.4% of matched S2/S3 records vs 0.3% of unmatched ones). An empty address therefore does not mean "no match"; for these pairs the model must rely on the name.',
   '<b>Duplicate names within S1 are common</b>: 30% of S1 rows repeat an exact raw name, and 50% share a normalised name (for example, 56 entities named "United Engineering Private Limited" across India).'])

# ---------- 5 Noise patterns
P('5. Noise patterns (observed examples)',h1)
P('Real S1 → S2/S3 pairs taken from the ground truth:')
T([['Pattern','S1 value','Matched S2/S3 value'],
 ['Case, prefix and legal suffix','CH Dynamic Auto Glass','The CH Dynamic Auto Glass · CH Dynamic Auto Glass  Inc.'],
 ['DBA / trade name','CH Dynamic Auto Glass','Rizatavo Co doing business as CH Dynamic Auto Glass'],
 ['Injected accents, punctuation, &','Bureau of Parks and Recreation','Bureau óf Parks and-Recreation · Bureau of Parks &amp; Recreation Authority'],
 ['Typos','Blue Alliance / First Networks Inc','Blue Alliatnae / FIRST NELWKSB INC'],
 ['Domain / handle as name','Phoenix Vidyalaya · Youth Unified Program','phoenixvidyalaya.com, ph0enixvidyalaya.com · #youthunified'],
 ['Junk tokens','Phoenix Vidyalaya','Phoenix  Center (ID: 47290) · Dr Phoenix Vídyalaya'],
 ['Word-order change','Jay It Private Limited','Private Jay It Limited'],
 ['Script change (India)','Premier Foundation Private Limited','[same name written in Devanagari]'],
 ['Unrelated name','White Nuclear Inc','Jaxorbi (identical address)'],
 ['House-number noise','11265 Sunrise Gold Circle','011265 SUNRISE GOLD CIRCLE · ##15140 ELDERFLOWER LANE'],
 ['Component reorder, full state','5212 Tinkers Creek Place, Clinton, MD','MD, 5212 TINKERS CREEK PLACE, CLINTON · …, Clnton, Maryland'],
 ['City alias / misspelling','Pune, Maharashtra · Penfield, NY','Poona, Pune, MH · Pennfield, New York'],
 ['State in native script','…, Karnataka','…, [Karnataka in Kannada script] / KA'],
 ['Partial / empty address','Sno 32/2/1 Hno 1048, Gulabnagar, …, Pune','BLOCK A-825 SNO 32/2/1 HNO 1048, PUNE, … · (empty)'],
 ['French abbreviations (test)','Rue de Dieppe, Lille','63 R. DE DIEPPE, LILLE · 51 R DE LA PLANCHE AU GUE']],[38,55,77])
T([['Rate of pattern among true pairs (200k sample)','India S2','India S3','US S2','US S3'],
   ['Raw name exactly equal','2.8%','2.7%','6.0%','5.9%'],['Normalised name exactly equal','42.7%','44.7%','56.7%','52.7%'],
   ['Name in Indic script','23.7%','13.2%','0%','0%'],['First house number equal','66.3%','65.3%','76.5%','77.4%'],
   ['Empty address','3.9%','4.1%','4.9%','4.6%'],['Domain-style name','3.5%','3.8%','4.6%','4.4%'],['DBA pattern','0.0%','1.6%','0.0%','2.1%']],[70,25,25,25,25])
IMG('namesim.png',160,'Figure 3. Name similarity of true pairs. Most US pairs score above 95. India has a heavy low tail, mostly Indic-script names that raw string metrics cannot compare. About 6% of non-Indic true pairs score below 60 (unrelated or domain names).')
T([['Similarity (true pairs)','India S2','India S3','US S2','US S3'],
   ['Name token-set, 5th percentile','5','6','60','59'],['Name token-set, median','100','100','100','100'],
   ['Address token-set, 5th percentile','60','43','53','54'],['Address token-set, median','95','91','93','86']],[70,25,25,25,25])

# ---------- 6 Hard negatives
P('6. Hard negatives: how distractors are built',h1)
P('To see what the matcher must reject, US S2/S3 records were joined to every US S1 entity with the same normalised name, and pairs with an address token-set similarity of at least 70 were kept (176,897 pairs, 82% of them true). The remaining 18% are near-copies that differ only in small details:')
T([['Distractor (S2)','Closest S1 (not its true entity)'],
   ["Schulman's Industries Center Corp · 18641 MANN LN, FAIRHOPE, AL","Schulman's Industries Center · 18634 Mann Lane, AL, Fairhope"],
   ['Golden Electronics Dynamics, Llc · 142 LINCOLN ST, LEOMINSTER, MA','Golden Electronics Dynamics, Corp · 135 Lincoln Street, Leominster, MA'],
   ['Mann Safe Activate Inc · 615 Eye St, HARLINGEN, TX','Mann Safe Activate LLC · 606 Eye Street, Harlingen, TX'],
   ['Bobby Banuelos Quality Diagnostics Ltd · 2813- YANCEYVILLE STREET','Bobby Banuelos Quality Diagnostics Inc · 2811 Yanceyville Street, Unit B']],[85,85])
IMG('housediff.png',170,'Figure 4. Among hard negatives the house-number difference is almost always exactly 1, 2, 3, 4, 5, 7, 9 or 11; the differences 6, 8, 10 and 12 almost never occur. This is the generator\'s fingerprint. True pairs keep the same number 87.7% of the time; their non-zero differences are mostly 1–2 (digit noise) or large (a unit or plot number parsed as the house number).')
IMG('signals.png',150,'Figure 5. Two cheap features separate the classes well: exact house-number equality and exact legal-suffix equality. Hard negatives almost always change the legal suffix (LLC → Corp, Inc → Ltd).')
KEY('<b>Implication.</b> A name-plus-address fuzzy score will rank these distractors as matches, and on a singleton a single false merge costs the full 1.0. Feature engineering must include: a robust house-number parser, the absolute difference and membership of the difference in {1,2,3,4,5,7,9,11}, and legal-suffix extraction with equality and compatibility flags. Note that a changed suffix alone is not decisive, since 35% of true pairs also change it.')

# ---------- 7 Blocking
P('7. Blocking (candidate generation) analysis',h1)
P('Within-country exact keys were evaluated against all 7.64M true training pairs. Addresses were lower-cased, accents removed, common street words abbreviated and leading zeros stripped.')
IMG('blocking.png',160,'Figure 6. Pair recall of simple exact keys. Even their union misses about 20% of true pairs.')
T([['Key (within country)','Pair recall','Candidate pairs','Per S1','Largest block'],
   ['Normalised name','50.5%','106.7M','48.3','970,700'],['House no. + first street word','48.7%','172.8M','78.3','8.9M'],
   ['House no. + two street words','38.9%','13.2M','6.0','543,099'],['First name token + house no.','52.6%','22.6M','10.2','191,700'],
   ['Union of all four','79.8%','—','—','—']],[60,25,30,22,33])
P('With the three best keys and a perfect matcher, the best achievable macro F0.5 is <b>0.909</b>. Predicting an empty list for every entity scores 0.056.')
KEY('<b>Implication.</b> The blocking stage caps recall and must combine several methods: (a) the exact keys above; (b) TF-IDF character n-gram (3–4) nearest neighbours on name + address within each country; (c) nearest neighbours on a small multilingual text encoder (MIT/Apache) to catch Indic-script and unrelated-name cases, which also generalises to France. Target at least 97% pair recall at about 20–50 candidates per S1, measured on a held-out split. The very large blocks (8.9M pairs) show that frequent-token keys need frequency caps.')

# ---------- 8 Preprocessing
P('8. Recommended preprocessing',h1)
P('Reading and cleaning',h2)
B(['Read with <font name="DVM">sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE</font>.',
   'Treat NULL, None, N/A and null inside addresses as empty tokens; add a flag for an empty address.',
   'Apply Unicode NFKD, strip combining accents and collapse repeated whitespace.'])
P('Names',h2)
B(['Lower-case; &amp; → "and"; remove junk prefixes and suffixes (--, ***, #, The, Dr, Smt, "(ID: n)", "| www…", brackets).',
   'Split DBA names ("X doing business as Y", "dba", "d/b/a", "t/a") and keep both parts.',
   'Recover words from domains and handles (firstnetworks.com → firstnetworks) and use them for character-level comparison.',
   '<b>Extract the legal suffix as a separate field</b> rather than just deleting it, and map variants to one form: Pvt/Private, Ltd/Limited, L.L.C./LLC, Corp/Corporation, Co/Company, plus SARL/SAS/SASU/EURL/SCI/SA for France.',
   'Transliterate Indic scripts to Latin, either with an MIT/Apache transliteration library or with a character mapping learned from the aligned training pairs (hundreds of thousands of Indic ↔ Latin name pairs).'])
P('Addresses',h2)
B(['Parse into house/plot number, street, unit, city, state and landmark. Strip leading zeros and "#"/"##" from numbers.',
   'Abbreviation tables per country: US (Street→St, Road→Rd …), India (H.No, Plot No, Opp, Nr), France (Rue→R, Boulevard→Bd, Avenue→Av, bis/ter).',
   'Map state names, codes and native-script state names to one canonical code, using a dictionary learned from the data.',
   'Build a city alias table from training pairs (Poona↔Pune, Calcutta↔Kolkata, "City of X"↔X, "X Township"↔X) and use typo-tolerant city matching.',
   'Compare addresses as sorted component sets, since component order varies.'])
P('Generalising to France',h2)
B(['Treat country as an open set. Every normalisation rule should be country-generic, with an optional country-specific table on top.',
   'Prefer features that do not depend on country (numeric equality, character n-gram similarity, embedding similarity) so the matcher transfers to unlabelled France.'])

# ---------- 9 Modelling
P('9. Suggested modelling pipeline',h1)
T([['Stage','Approach'],
   ['Blocking','Union of exact keys, TF-IDF character n-gram kNN and multilingual-embedding kNN (FAISS on GPU), all within country; cap block sizes. Write the final union to candidate_pairs.tsv.'],
   ['Pair features','Name: token-set / Jaro-Winkler / TF-IDF cosine, legal-suffix equal, domain and DBA flags, name frequency in S1. Address: house number equal, absolute difference, difference in {1,2,3,4,5,7,9,11}, street / city / state / unit similarity, empty flags. Source (S2/S3), country, embedding cosine.'],
   ['Matcher','Gradient-boosted trees (LightGBM/XGBoost) trained on candidate pairs, with hard negatives taken from the blocking output. Optionally, a small fine-tuned cross-encoder for uncertain pairs.'],
   ['Post-processing','Assign each S2/S3 record only to its best-scoring S1 (the ground truth is one-to-many). Tune the threshold for macro F0.5 on validation. Add transitive support: an S3 record almost identical to an accepted S2 record is also accepted.'],
   ['Validation','Hold out about 10% of S1 entities with all their candidates, compute macro F0.5 exactly as specified, and report blocking recall and reduction ratio separately.']],[30,140])

# ---------- Appendix
P('Appendix: method notes',h1)
B(['Environment: Python 3.11 (conda env "general"), pandas, rapidfuzz; 40 CPUs, 500 GB RAM, 4× V100.',
   'Name normalisation: NFKD accent removal, lower-case, &amp;→and, punctuation removed, "(ID: n)" removed, legal and generic tokens dropped (inc, llc, ltd, pvt, private, corp, co, the, holdings, group, sarl, sas …).',
   'Similarity statistics use a 200,000-pair random sample of true pairs (seed 0). The hard-negative analysis uses a 400,000-record US sample of S2 ∪ S3 joined to S1 on normalised name.',
   'House number is the first integer in the address string after normalisation, which explains the long tail of large differences for true pairs where a unit or plot number comes first.',
   'Examples shown in this report are verbatim records from the training data. Indic-script strings are described in words because the report font lacks those glyphs.'])

def deco(c,d):
    c.saveState(); c.setFont('DV',7.5); c.setFillColor(SUB)
    c.drawString(20*mm,12*mm,'Amazon ML Challenge 2026 · EDA'); c.drawRightString(190*mm,12*mm,f'{d.page}'); c.restoreState()
doc=SimpleDocTemplate('/home/user/amazon_ml/EDA.pdf',pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=18*mm,bottomMargin=20*mm,
    title='EDA — Amazon ML Challenge 2026 Entity Resolution',author='bhuvanpalyam@iisc.ac.in')
doc.build(S,onFirstPage=deco,onLaterPages=deco); print('built')
