from hparams import hp
import json
from tqdm import tqdm
from collections import Counter
from operator import itemgetter
import os

def normalize(text):
    text = text.strip(" -\n")
    return text

if __name__ == "__main__":
    #Agrupar frases
    en2pts = dict()
    en_lines = open(hp.opus_en, 'r').read().splitlines()
    pt_lines = open(hp.opus_pt, 'r').read().splitlines()
    for pt, en in tqdm(zip(pt_lines, en_lines), total=len(pt_lines)):
        pt = normalize(pt)
        en = normalize(en)
        if len(pt) <=1: continue
        if en not in en2pts: en2pts[en] = []
        en2pts[en].append(pt)
    print(f"Todas as frases sinõnimos foram agrupadas: {len(en2pts)}")

    # Ordenação
    data = dict()
    i = 0
    for en, pts in en2pts.items():
        pt2cnt = Counter(pts)
        phrases = sorted(pt2cnt.items(), key=itemgetter(1), reverse=True)
        if len(phrases) > 1:
            val = dict()
            val["_translation"] = en
            val["phrases"] = phrases
            data[i] = val
            i += 1
    print(f"Sorted all synonymous groups by frequency: {len(data)}")

    # Write
    os.makedirs(os.path.dirname(hp.sg), exist_ok=True)
    with open(hp.sg, 'w') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(',', ': '), sort_keys=True)