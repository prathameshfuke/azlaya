import json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

OOV = json.load(open('oov.json')); OOV2 = json.load(open('oov2.json')); TL = json.load(open('translit.json'))
BLUE, ORANGE, AQUA, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e6e5e1'
os.makedirs('pfig', exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.dpi': 200, 'savefig.bbox': 'tight'})

# Figure A: share of test token occurrences never seen in train
lab = ['US names', 'India names\n(Latin)', 'India names\n(Indic words)', 'France names', 'France\naddresses']
ind_unseen = np.average([v['unseen_occ'] for v in OOV2['business_name'].values()], weights=[v['test_occ'] for v in OOV2['business_name'].values()])
val = [OOV['us_name_unseen_occ'], OOV['india_ascii_name_unseen_occ'], ind_unseen, OOV['france_name_unseen_occ'], OOV['france_addr_unseen_occ']]
val = [v * 100 for v in val]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.bar(range(5), val, color=[BLUE, BLUE, BLUE, ORANGE, ORANGE], edgecolor='white', linewidth=1.5, width=0.62)
for i, v in enumerate(val): ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=8, color=INK)
ax.set_xticks(range(5), lab); ax.grid(axis='x', visible=False); ax.set_ylabel('% of test tokens')
ax.set_title('Test tokens never seen in training (S2+S3, by occurrence)', loc='left', color=INK, fontsize=10, fontweight='bold')
fig.savefig('pfig/oov.png'); plt.close(fig)

# Figure B: transliteration ladder
steps = [('No transliteration', 'raw (no transliteration)'), ('Rule-based IAST', 'IAST rule-based'), ('+ phonetic simplify', 'IAST + phonetic simplify'),
         ('+ schwa & anusvara rules', 'IAST + simplify + schwa/anusvara'), ('+ consonant skeleton', '... + consonant skeleton')]
v = [TL[k]['char_ratio_median'] for _, k in steps]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.barh([s for s, _ in steps], v, color=[BLUE] * 4 + [AQUA], edgecolor='white', height=0.6); ax.invert_yaxis(); ax.set_xlim(0, 100)
for i, x in enumerate(v): ax.text(x + 1, i, f'{x:.1f}', va='center', fontsize=8, color=INK)
ref = TL['latin_india_char_ratio_median']; ax.axvline(ref, color=INK2, lw=1)
ax.text(ref - 1, 0, f'Latin-script India pairs: {ref:.1f}', ha='right', va='center', fontsize=7.5, color=INK2)
ax.grid(axis='y', visible=False); ax.set_xlabel('Median character similarity to the S1 English name (0-100)')
ax.set_title(f'Indic-script names: effect of each transliteration step ({OOV2.get("n_pairs", 14669):,} true pairs)', loc='left', color=INK, fontsize=10, fontweight='bold')
fig.savefig('pfig/translit.png'); plt.close(fig)

# ---------------------------------------------------------------- PDF
F = '/usr/share/fonts/truetype/dejavu/'
for n, f in (('DV', 'DejaVuSans.ttf'), ('DVB', 'DejaVuSans-Bold.ttf'), ('DVI', 'DejaVuSans-Oblique.ttf'), ('DVM', 'DejaVuSansMono.ttf')):
    pdfmetrics.registerFont(TTFont(n, F + f))
registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')
C_INK, C_SUB, C_ACC, C_LINE, C_BG = colors.HexColor(INK), colors.HexColor(INK2), colors.HexColor(BLUE), colors.HexColor('#d9d8d3'), colors.HexColor('#f4f3ef')
body = ParagraphStyle('b', fontName='DV', fontSize=9.2, leading=13.2, textColor=C_INK, spaceAfter=5)
bul = ParagraphStyle('bu', parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
h1 = ParagraphStyle('h1', fontName='DVB', fontSize=14, leading=18, textColor=C_INK, spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle('h2', fontName='DVB', fontSize=11, leading=14, textColor=C_INK, spaceBefore=10, spaceAfter=3, keepWithNext=1)
lab_s = ParagraphStyle('lab', fontName='DVB', fontSize=8.2, leading=11, textColor=C_ACC, spaceBefore=3, spaceAfter=1, keepWithNext=1)
cap = ParagraphStyle('c', parent=body, fontSize=8, leading=11, textColor=C_SUB, spaceAfter=10)
cell = ParagraphStyle('cell', fontName='DV', fontSize=7.8, leading=10, textColor=C_INK)
cellb = ParagraphStyle('cellb', parent=cell, fontName='DVB')
code = ParagraphStyle('code', fontName='DVM', fontSize=7.6, leading=10, textColor=C_INK)
title = ParagraphStyle('t', fontName='DVB', fontSize=22, leading=27, textColor=C_INK)
S = []
def P(t, s=body): S.append(Paragraph(t, s))
def BUL(items):
    for t in items: S.append(Paragraph(t, bul, bulletText='•'))
def T(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, -1), 0.4, C_LINE), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 3),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('BACKGROUND', (0, 0), (-1, 0), C_BG), ('LINEBELOW', (0, 0), (-1, 0), 0.8, C_SUB)]))
    S.append(t); S.append(Spacer(1, 8))
def IMG(fn, w=160, c=None):
    iw, ih = ImageReader(fn).getSize(); S.append(Image(fn, width=w * mm, height=w * mm * ih / iw))
    P(c, cap) if c else S.append(Spacer(1, 8))
def BOX(t):
    k = Table([[Paragraph(t, ParagraphStyle('k', parent=body, spaceAfter=0))]], colWidths=[170 * mm])
    k.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), C_BG), ('LINEBEFORE', (0, 0), (0, -1), 2.5, C_ACC), ('LEFTPADDING', (0, 0), (-1, -1), 8),
                           ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    S.append(k); S.append(Spacer(1, 8))
def CODE(lines):
    k = Table([[Paragraph(l.replace(' ', '&nbsp;'), code)] for l in lines], colWidths=[170 * mm])
    k.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), C_BG), ('LEFTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1)]))
    S.append(k); S.append(Spacer(1, 8))
def METHOD(n, name, what, why, how, unseen, variants=None):
    """One preprocessing method: what / why (evidence) / how / unseen-word behaviour / alternatives."""
    P(f'{n} {name}', h2)
    P('What it does', lab_s); P(what)
    P('Why it helps', lab_s); P(why)
    P('How', lab_s); BUL(how) if isinstance(how, list) else P(how)
    P('Unseen words in test', lab_s); P(unseen)
    if variants:
        P('Alternatives and options', lab_s); BUL(variants)
pct = lambda x, d=1: f'{x * 100:.{d}f}%'

# ---------------------------------------------------------------- content
S.append(Spacer(1, 6)); P('Preprocessing Strategy', title)
P('Amazon ML Challenge 2026 — Business Entity Resolution', ParagraphStyle('st', parent=body, fontSize=12, leading=16, textColor=C_SUB))
P('Companion to EDA.pdf. It describes each preprocessing method, why it helps on this data, and how it behaves on words that appear in the test set but not in training. All figures were measured on the provided data.', cap)
S.append(HRFlowable(width='100%', color=C_LINE, thickness=0.8, spaceAfter=8))

P('Summary', h1)
BUL(['<b>Parse, don\'t just clean.</b> Turn each record into structured fields (core name, legal form, house number, unit, street, city, state) plus a few normalised strings. The hard negatives differ from true matches only in details that plain cleaning deletes, mainly the house number and the legal suffix.',
     '<b>For words never seen before, rules and character-level methods beat word lists.</b> ' + f'{pct(OOV["france_name_unseen_occ"], 0)} of French name tokens and {pct(OOV["france_addr_unseen_occ"], 0)} of French address tokens in the test set never occur in training. A method that works only through a list of known words will fail on them. Rule-based transliteration, character n-grams and phonetic keys still work.',
     '<b>Indic scripts use a closed vocabulary.</b> Each script has only about 165–176 distinct words, and every Indic word in the test set also appears in training. A word dictionary learned from the training pairs therefore covers Indic text completely. Rule-based transliteration with phonetic folding is the safety net: on its own it takes the median character similarity of Indic-script names to their English S1 name from ' + f'{TL["raw (no transliteration)"]["char_ratio_median"]:.0f} to {TL["... + consonant skeleton"]["char_ratio_median"]:.1f}, slightly above Latin-script India pairs ({TL["latin_india_char_ratio_median"]:.1f}), with no word list.',
     '<b>France is where unseen words matter.</b> There are no French training records, and the legal forms (SARL, SAS, EURL …), street types (rue / r / r., av, bd, allée) and region / department names must be handled by rules, character-level similarity and statistics fitted on the test files.',
     '<b>Use dictionaries learned from training as the first layer</b> (Indic words, legal forms, city and state aliases), and always fall back to the rule-based path for anything not in the dictionary.',
     '<b>Measure every rule against the ground truth.</b> A useful rule raises field agreement on true pairs without raising it on hard negatives. Use held-out words or a held-out country to check that it generalises.'])

P('1. How different is the test vocabulary?', h1)
P('Tokens from the S2+S3 test files were compared with the S2+S3 training vocabulary, weighted by how often they occur.')
IMG('pfig/oov.png', 160, 'Figure 1. US and India Latin-script names are mostly known words (the unseen share is largely injected typos and random brand names). France is the real gap: about two in five name tokens are new words (école, amicale, sàrl, établissements, Tourcoing …).')
rows = [['Script (India names, word tokens)', 'Distinct words in train', 'Test occurrences', 'Unseen in train']]
for s, v in sorted(OOV2['business_name'].items(), key=lambda x: -x[1]['test_occ']):
    rows.append([s, f"{v['train_types']:,}", f"{v['test_occ']:,}", pct(v['unseen_occ'], 2)])
T(rows, [55, 40, 40, 35])
P(f"Of India test names that contain Indic script, {pct(OOV2['rows_with_unseen_indic'])} contain at least one Indic word never seen in training. "
  f"Names are {pct(OOV2['full_vs_mixed'].get('full', 0), 0)} fully in Indic script and {pct(OOV2['full_vs_mixed'].get('mixed', 0), 0)} code-mixed (for example an English word followed by an Indic one).", body)
BOX('<b>What this means.</b> Non-English text is two different problems. <b>Indic text is closed:</b> the data generator uses a fixed list of about 170 words per script, all of which appear in training, so a dictionary from Indic word to English word (learned by aligning true pairs) gives an exact translation for every test token. <b>French text is open:</b> about 39% of name tokens and 23% of address tokens are new, so it needs rules, character-level similarity and statistics computed on the test files. Keep the rule-based transliteration as well: it costs little and protects against any Indic word the dictionary misses.')
P('A tokenisation pitfall found during this analysis: Python\'s <font name="DVM">\\w</font> regex does not match Indic vowel signs (matras), so <font name="DVM">re.findall(r"\\w+")</font> splits Indic words into syllable fragments. Tokenise Indic text on whitespace and punctuation instead.', cap)

P('2. Text-level preprocessing (both fields)', h1)
METHOD('2.1', 'Safe loading and missing-value handling',
       'Reads every file as text, so no value is silently turned into NaN. Recognises placeholder tokens inside addresses.',
       'Names such as "NULL" or "NA" would otherwise become missing values. 2.5% of S2/S3 addresses contain a literal NULL, None or N/A, and 3–5% are empty. Empty addresses are 15× more common among matched records (4.4%) than among unmatched ones (0.3%), so "no address" must be modelled rather than treated as "no match".',
       ['<font name="DVM">pd.read_csv(p, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)</font>',
        'Remove NULL / None / N/A / null / nan as tokens inside addresses; add <font name="DVM">addr_empty</font> and <font name="DVM">addr_n_components</font> features.',
        'Keep the raw strings unchanged next to the parsed fields, for character-level features and error analysis.'],
       'Not word-dependent; works the same on any country.')
METHOD('2.2', 'Unicode normalisation and accent folding',
       'Applies NFKC (unifies look-alike characters and full-width forms), then NFKD and removes combining marks. This maps é→e, à→a, ç→c.',
       'The generator injects accents into English names ("Bureau óf Parks", "Vídyalaya"). French text is written both with and without accents (allée / allee, Mérignac / Merignac both appear in test). Without folding these count as different tokens.',
       ['Apply to <b>Latin script only</b>. For Indic text use NFC and keep the combining marks: the vowel signs are combining characters, and stripping them destroys the word.',
        'Keep an accent-preserving copy for the embedding model, which can use the accents.'],
       'Fully general: works on any Latin-script word, seen or unseen. This is the main defence for French accents.',
       ['<font name="DVM">unidecode</font> (GPL — check the licence) or <font name="DVM">anyascii</font> (ISC) as a broader ASCII folding, also covering ligatures such as œ→oe.'])
METHOD('2.3', 'Case, punctuation and script-aware tokenisation',
       'Lower-cases, maps &amp;→"and" and "+"→"plus", replaces punctuation with spaces, and collapses whitespace. Tokenises Latin text with a regex and Indic text on whitespace and punctuation.',
       'S2 US addresses are 93% upper-case, and names contain "and-Recreation", "CORNERSTONE-GAMING", "L.L.C.". Punctuation inside number tokens carries meaning (India "32/2/1", "B-16", "H.No"), so it must be protected.',
       ['Keep "/" and "-" when they sit between digits or letter+digit; otherwise replace them with a space.',
        'Join dotted abbreviations before removing dots: "L.L.C." → "llc", "S.A.R.L." → "sarl", "Pvt." → "pvt".',
        'Remove repeated letters only for comparison keys (see 2.6), never in the stored field.'],
       'General. The dotted-abbreviation rule covers any unseen acronym ("S.C.O.P." → "scop").')
METHOD('2.4', 'Transliteration of Indic scripts',
       'Converts Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam into Latin letters, so they can be compared with the English S1 name.',
       '24% of India names in S2 and 13% in S3 are in Indic script (plus native-script state names in addresses). Without transliteration their similarity to S1 is zero, and they would be missed or matched only on address.',
       ['<b>Layer 0 (main path for this data): a learned word dictionary.</b> Indic text uses only about 170 distinct words per script, and every test Indic word appears in training. Align Indic and English tokens in true training pairs (same position after removing legal words, or the best character-similarity match after transliteration) and keep the most frequent English word for each Indic word (e.g. the Devanagari spellings of "private" and "limited" map back to those English words). This gives exact English words, which is better than any transliteration.',
        '<b>Layer 1 (fallback): rule-based transliteration</b> with <font name="DVM">indic-transliteration</font> (MIT licence), per token, detecting the script from its Unicode block: <font name="DVM">sanscript.transliterate(tok, SCRIPT, sanscript.IAST)</font>.',
        '<b>Layer 2: script-specific cleanup.</b> Delete the final inherent vowel ("limiteda" → "limited", common in Kannada and Telugu output), and write the nasal mark (anusvara) as "n" before a consonant ("imfotek" → "infotek").',
        '<b>Layer 3: phonetic folding</b>, applied to <i>both</i> sides: collapse doubled letters, w→v, sh→s, ph→f, z→j, then optionally a consonant skeleton (drop non-initial vowels: "praivet" → "prvt", "private" → "prvt").',
        'Apply the same method to native-script state names in addresses (1–8 forms per script, e.g. Maharashtra written in Devanagari → Maharashtra → MH).'],
       'On this data, no Indic word in the test set is unseen, so the dictionary covers everything. The rule-based layers need no vocabulary and act as insurance: if the hidden test portion contains a new word, it still gets a close Latin spelling instead of being dropped.',
       ['Learn a character-level transliteration model (a small seq2seq or a weighted finite-state transducer) from the aligned training pairs. It captures this data\'s English-to-Indic spelling habits better than IAST, and still works on unseen words.',
        'Transliterate the <i>English</i> S1 name <i>into</i> each Indic script and compare in that script. This is useful as a second, independent view.',
        'Rely on a multilingual encoder (e.g. multilingual-e5-small, MIT) that already understands Indic scripts. It is good for blocking but not precise enough to be the only signal.'])
IMG('pfig/translit.png', 160, 'Figure 2. Median character similarity between an Indic-script S2/S3 name and its true English S1 name, after each step. Each rule adds on top of the previous one; with all of them, Indic pairs are about as easy as Latin-script India pairs (reference line).')
T([['English S1 name (folded)', 'Indic S2/S3 name after all layers'],
   ['digital infotech private limited', 'dijital infotek praivet limited'],
   ['gold consulting private limited', 'gold kansalting praivet limited'],
   ['svastik energy limited', 'svastik enarji limited'],
   ['south products private limited', 'saut prodakts praivet limited']], [85, 85])
P('The remaining differences (praivet / private, enarji / energy) are systematic spelling habits. The layer-0 dictionary removes them entirely for known words; for unknown words, the consonant skeleton and character n-grams absorb them.', cap)
METHOD('2.5', 'Typo robustness: character n-grams and phonetic keys',
       'Represents each string by overlapping character sequences (3–4 characters) and by a phonetic code, so a few wrong letters change the representation only slightly.',
       'Typos are injected on every side: "Alliatnae", "NELWKSB", "Clnton", "Mayvlile", "Pllce". 77% of distinct Latin name tokens in the test set are unseen, mostly because of these typos; by occurrence, though, they are only about 7%.',
       ['TF-IDF on character 3–4-grams (<font name="DVM">analyzer="char_wb"</font>) for names and addresses. Fit IDF on train and test text together (unsupervised, using only the provided data).',
        'Double Metaphone for Latin-script tokens (MIT <font name="DVM">metaphone</font> or <font name="DVM">jellyfish</font>); the consonant skeleton for transliterated text.',
        'Store a <font name="DVM">name_sorted</font> key (tokens sorted) for word-order changes ("Private Jay It Limited").'],
       'Character n-grams are the main tool for unseen words: a new word shares most of its n-grams with its misspelled or accented variants, even though the word itself was never seen.',
       ['Typo-tolerant token matching (edit distance ≤1–2 between tokens) as an alternative to n-grams for short names.',
        'Subword (BPE / SentencePiece) tokenisation trained on train+test text, used as input to an embedding model.'])

P('3. Business-name preprocessing', h1)
METHOD('3.1', 'Junk prefix and suffix removal',
       'Removes injected tokens that carry no identity: leading "--", "***", "#", "&lt;&lt;", "The", "Dr", "Smt"; trailing "(ID: 47290)", "| www.x.com"; and brackets around legal words ("[Limited]", "((LLC))").',
       'These tokens appear throughout S2/S3 and lower both exact-match and token-overlap scores for true pairs. Together with the rest of the name normalisation (2.2, 2.3, 3.2), removing them raises the exact-match rate of names in true pairs from 3–6% (raw strings) to 43–57%.',
       ['A small list of patterns, applied only at the <i>start</i> or <i>end</i> of the name, so that real words such as "Dr Reddy Labs" are not damaged in the middle.',
        'Learn the list from data: tokens that appear at the start or end of S2/S3 names in true pairs but not in the paired S1 name.'],
       'The patterns are symbols and position rules rather than vocabulary, so they apply to any name. For France, "Sté" and "Ets" (société, établissements) can be added as optional generic words.')
METHOD('3.2', 'Legal-suffix extraction and canonical form',
       'Moves the legal form into its own field, <font name="DVM">legal_form</font>, mapped to a canonical value, and removes it from <font name="DVM">name_core</font>.',
       'This is one of the two strongest precision signals: hard negatives keep the same legal form only 1.5% of the time, true pairs 65% of the time. Deleting the suffix, as generic cleaning does, throws that signal away. Keeping it out of <font name="DVM">name_core</font> also stops "Pvt Ltd" from inflating name similarity.',
       ['US: inc/incorporated, llc/l.l.c., corp/corporation, co/company, pc, pllc, lp, llp. India: pvt/private, ltd/limited, llp, "private limited" as one unit. France: sarl / s.a.r.l / sàrl, sas / s.a.s., sasu, eurl, sci, sa / s.a., snc, association/asso.',
        'Compare as: equal / compatible (e.g. "private limited" vs "limited") / conflicting / missing on one side.',
        'Handle position: legal words may appear at the start ("PVT. EFS Print Ventures Ltd.", "SA PG Finance").'],
       'Legal forms are a closed set per country, so a small hand-written list per country covers the test set. For France all the variants above appear in the test data and none in training, which is why they must be added by hand.',
       ['Group legal forms by country so that a French "SA" is not confused with an English word.',
        'Keep a "generic words" list (holdings, group, services, enterprises, trust) separate from legal forms, because it behaves differently.'])
METHOD('3.3', 'DBA / trade-name splitting',
       'Splits "X doing business as Y", "X dba Y", "d/b/a", "t/a", "trading as" into two names and compares each with the S1 name.',
       'About 2% of S3 true pairs use this pattern (e.g. "Rizatavo Co doing business as CH Dynamic Auto Glass"). One half is usually a random brand, so comparing the whole string gives a low score for a true match.',
       ['Take the maximum similarity over the parts; add a <font name="DVM">has_dba</font> flag.'],
       'Works for any name. Add French equivalents (« exerçant sous le nom », « enseigne ») and Hindi ones as a small list if they appear.')
METHOD('3.4', 'Domain and handle reconstruction',
       'Turns "firstnetworks.com", "ph0enixvidyalaya.com", "#youthunified" or "*** nathholdings.com" back into comparable text.',
       'About 4% of true pairs use a domain or handle as the name. Without reconstruction they score below 50 on name similarity.',
       ['Drop the TLD (.com, .in, .net, .org, .fr, .co) and the leading #, @ or ***; map look-alike digits (0→o, 1→l, 3→e).',
        'Compare the result with the S1 core name <i>with spaces removed</i> ("firstnetworks" vs "first networks" → "firstnetworks").',
        'Optionally split the joined string into words using the S1 vocabulary (word segmentation).'],
       'Comparing without spaces needs no vocabulary. Word segmentation depends on known words, so use it only as an extra feature.')
METHOD('3.5', 'Name frequency and generic-word weighting',
       'Counts how many S1 entities share each core name and each token, and weights tokens by rarity (IDF).',
       '50% of S1 entities share their normalised name with another S1 entity (for example 56 "United Engineering Private Limited"). A match on a common name or on generic words such as "services" or "enterprises" is weak evidence; a match on a rare token is strong.',
       ['Fit counts on the test S1 file itself at inference time. The collisions that matter are the ones in the test set.',
        'Features: <font name="DVM">name_freq_s1</font>, IDF-weighted token overlap, the rarest shared token\'s IDF.'],
       'IDF computed on test text gives a weight to every test word, including unseen ones. Nothing is looked up.')

P('4. Address preprocessing', h1)
METHOD('4.1', 'House-number and unit parsing',
       'Extracts the street number (removing leading zeros and "#"/"##", handling "H.No", "Plot No", "S No", "Door No", "Gat No", "bis"/"ter") and, separately, the unit (Unit, Apt, Suite, Fl, Flat No, Shop No, Tower).',
       'The single most important field. True pairs share the house number 87.7% of the time; hard negatives differ by exactly 1–5, 7, 9 or 11 in 62% of cases. With the naive rule (first number in the string), 11% of true pairs still show a large difference, most likely because a unit or plot number comes first. A proper parser should remove most of these.',
       ['Store it as a string (e.g. "32/2/1", "b-16") and as an integer (for the difference features).',
        'Normalise "5 bis", "5bis", "5 b" → "5bis" (France); "011265" → "11265"; "##15140" → "15140".',
        'Features: equal, absolute difference, difference ∈ {1,2,3,4,5,7,9,11}, missing on one side, unit equal.'],
       'Numbers are language-independent, so this is the most robust signal for France. The keywords ("H.No", "Unit", "bis") come from a short list per country.',
       ['Train a small sequence tagger (CRF or a tiny transformer) that labels address tokens as number / unit / street / city / state, using the S1 addresses (which are clean and consistently ordered) as weak labels.'])
METHOD('4.2', 'Street-type abbreviation to one form',
       'Maps street words and their abbreviations to one form per country.',
       'The same street is written "Road" / "Rd" / "RD", "Place" / "Pl", "Rue" / "R" / "R.", "Avenue" / "Av" / "Av." / "Ave". In test France S2, "r" and "r." together (25% of street types) are almost as common as "rue" (37%).',
       ['US: street→st, road→rd, drive→dr, avenue→ave, lane→ln, boulevard→blvd, court→ct, place→pl, circle→cir, trail→trl, highway→hwy, parkway→pkwy, terrace→ter.',
        'India: nagar/ngr, road/rd, opposite/opp, near/nr, colony/col, sector/sec, house number/h.no, plot/pl no.',
        'France: rue/r, avenue/av/ave, boulevard/bd/boul, allée/all, impasse/imp, chemin/che/ch, place/pl, route/rte, quai, cours, faubourg/fbg.',
        'Also: saint/st/ste, north/n, first/1st.'],
       'The street types are a closed list per country, so a hand-written table covers unseen test data. Street <i>names</i> (Maréchal Foch, Jules Verne) are open vocabulary and are handled by the character-level similarity in 2.5.')
METHOD('4.3', 'City, state and region normalisation',
       'Maps state codes, full names and native-script names to one code, and city aliases and misspellings to one city.',
       'S1 uses state codes ("MD"), S3 full names ("Maryland"), S2 native script (Maharashtra written in Devanagari). Cities appear as aliases (Poona/Pune, Calcutta/Kolkata, "City of Mayville", "Concord Township") and misspelled (Clnton, Mayvlile). <b>In test France, S1 names the region (Hauts-de-France in 39% of S1 addresses) while S2/S3 often name the department instead (Nord, Gironde, Loire-Atlantique in about 10% each).</b>',
       ['States: code ↔ name ↔ native script, learned from aligned training pairs (US 50 states, India 36 states and union territories).',
        'City aliases: count the city pairs that differ between S1 and S2/S3 in true training pairs, and keep frequent ones.',
        'Typo-tolerant city matching against the list of cities seen in S1 (edit distance ≤2, or char n-gram similarity).',
        '<b>France department ↔ region without labels:</b> learn it from the test files themselves. The same city (e.g. Tourcoing) appears with "Hauts-de-France" in S1 and with "Nord" in S2/S3, so co-occurrence counts link each department to a region. This is unsupervised and uses only provided data.'],
       'The learned alias tables cover only what was seen in training. For France, use the co-occurrence method above plus typo-tolerant matching. A tiny hand-written department → region table is an alternative, but check it against the external-data rule first.')
METHOD('4.4', 'Component-order invariance',
       'Compares addresses as sets of components rather than as ordered strings.',
       'Component order is shuffled on purpose ("TX, Frisco, 15140 Elderflower Lane" vs "15140 Elderflower Lane, Frisco, TX"; "Pune, Maharashtra, …, Sno 32/2/1"). Ordered string similarity penalises these true pairs.',
       ['Split on commas, classify each component by type (number, unit, street, city, state, landmark), and compare component by component; also keep a sorted token set.'],
       'Not word-dependent.')
METHOD('4.5', 'India landmarks and partial addresses',
       'Separates landmark phrases ("Near SBI ATM", "Opp Panjarpol", "Behind M.C Quarters", "Next to Iter College") and marks addresses that are partial.',
       '7–11% of India addresses contain a landmark, and records often drop some components (the state, the locality, or everything except the city). Landmarks are useful when both sides have them, but they add noise to overall address similarity.',
       ['Move landmarks into a separate field and compare them separately.',
        'Features: number of components on each side, share of the shorter address\'s tokens found in the longer one (containment rather than symmetric similarity).'],
       'The trigger words (near, opp, behind, next to, beside) are a small closed list; the landmark text itself is compared character by character.')

P('5. Representations for blocking and matching', h1)
METHOD('5.1', 'Blocking strings and keys',
       'Builds, per record, the strings used by each blocking method: <font name="DVM">name_core</font> (transliterated), <font name="DVM">house_no + street + city</font>, and a combined "name | address" string for the embedding model; always within country.',
       'Simple exact keys reach only 79.8% pair recall. Transliteration, junk removal and consistent abbreviations each raise the recall of exact and fuzzy keys, so preprocessing directly raises the recall ceiling.',
       ['Exact keys: core name; house number + first street word; first name token + house number; consonant skeleton of core name.',
        'Fuzzy: TF-IDF character n-gram nearest neighbours; multilingual embedding nearest neighbours.'],
       'The skeleton key, character n-grams and a multilingual embedding all work on unseen words. Exact word keys do not, so never rely on them alone.')
METHOD('5.2', 'Multilingual embeddings',
       'Encodes the combined string with a small multilingual sentence encoder (MIT/Apache, well under the 8B limit), for nearest-neighbour blocking and as a similarity feature.',
       'The encoder places an Indic-script name and its English form close together and understands French, which the training data does not cover. It also catches the ~6% of true pairs whose names are unrelated, through the address.',
       ['Feed it the <i>lightly</i> normalised string (keep accents and native script); the heavy folding above is for the string features.',
        'Optionally fine-tune it with contrastive learning on training pairs, using the hard negatives from EDA §6.'],
       'This is the strongest general method for unseen words, because the encoder was pre-trained on large amounts of multilingual text. Fine-tuning on US and India should not remove its French knowledge if learning is kept gentle.')

P('6. Strategy for words not seen in training', h1)
P('Each comparison should go through a layered fallback, so a word that is missing from one layer is still handled by the next:')
T([['Layer', 'Handles', 'Works on unseen words?'],
   ['1. Learned dictionaries (Indic words → English, legal forms, city / state aliases)', 'Known variants, with high precision; covers 100% of test Indic words', 'No — known words only'],
   ['2. Rules (Unicode folding, transliteration, schwa / anusvara, abbreviation tables)', 'Systematic variation', 'Yes'],
   ['3. Character n-grams, edit distance, phonetic codes, consonant skeleton', 'Typos and spelling habits', 'Yes'],
   ['4. Statistics fitted on test text (IDF, name frequency, co-occurrence alias tables)', 'Word importance, France geography', 'Yes — computed on test'],
   ['5. Multilingual embeddings', 'Cross-script meaning, unrelated names', 'Yes']], [80, 50, 40])
P('How to check that it generalises, before submitting:', lab_s)
BUL(['<b>Held-out words:</b> remove 30% of the learned dictionary entries, rerun, and confirm that matching quality drops only slightly. That shows the rule-based layers carry the load.',
     '<b>Held-out country:</b> build the pipeline using US only, validate on India (and the reverse). This is the closest available test of how well it will transfer to France.',
     '<b>Parse rate on test France:</b> the share of France records whose house number, street, city and legal form are all filled. It should be similar to the US and India parse rates; a large gap points to a missing rule.',
     '<b>Features must not identify specific words.</b> Use similarity scores and flags rather than token IDs or one-hot words, and never one-hot the country, so the model transfers to unseen vocabulary.'])

P('7. Recommended order of work', h1)
T([['#', 'Step', 'Main benefit', 'Check'],
   ['1', 'Safe loading, Unicode folding, tokenisation (2.1–2.3)', 'Foundation', 'Row counts; no NaN names'],
   ['2', 'House-number and unit parser (4.1)', 'Precision (hard negatives)', 'House number equal on >95% of US true pairs; <1% of hard negatives'],
   ['3', 'Legal-suffix extraction, all three countries (3.2)', 'Precision', 'Equality rate: true pairs ≈65%, hard negatives ≈1.5%'],
   ['4', 'Junk / DBA / domain handling (3.1, 3.3, 3.4)', 'Recall', 'Core name exact match on true pairs'],
   ['5', 'Indic word dictionary + rule-based transliteration fallback (2.4, 2.5)', 'Recall (India)', 'Dictionary coverage of test Indic words (100% expected); median similarity of Indic pairs ≈ Latin pairs'],
   ['6', 'Street / city / state normalisation, France department ↔ region (4.2–4.4)', 'Recall and precision', 'Parse rate on test France'],
   ['7', 'Blocking strings, IDF on test, embeddings (5.1, 5.2)', 'Recall ceiling', 'Blocking recall on validation ≥97%']], [6, 64, 38, 62])
P('Implementation note: about 28M records need parsing. Run the parser once with multiprocessing over chunks (40 cores), store every derived field as parquet, and reuse it; do not parse inside the feature loop.', cap)

P('Appendix: measurements behind this report', h1)
BUL([f'Vocabulary comparison: S2+S3 test tokens vs S2+S3 train tokens, by occurrence. Latin text tokenised with a word regex after lower-casing; Indic words split on whitespace and punctuation (India records only).',
     f'Transliteration: all {OOV2.get("n_pairs", 14669):,} Indic-script true pairs in the 200,000-pair EDA sample; metric = median rapidfuzz ratio between the folded S1 name and the processed S2/S3 name. Library: indic-transliteration 2.3 (MIT), IAST scheme.',
     'France counts: test S1/S2/S3 France records (259k / 703k / 732k).',
     'Hard-negative and field-agreement numbers are from EDA.pdf (code/eda.py).'])

def deco(c, d):
    c.saveState(); c.setFont('DV', 7.5); c.setFillColor(C_SUB)
    c.drawString(20 * mm, 12 * mm, 'Amazon ML Challenge 2026 · Preprocessing'); c.drawRightString(190 * mm, 12 * mm, f'{d.page}'); c.restoreState()
doc = SimpleDocTemplate('/home/user/amazon_ml/Preprocessing.pdf', pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                        bottomMargin=20 * mm, title='Preprocessing Strategy — Amazon ML Challenge 2026')
doc.build(S, onFirstPage=deco, onLaterPages=deco); print('built')
