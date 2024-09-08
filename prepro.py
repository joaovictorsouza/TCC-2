from hparams import hp
import pickle, os
from tqdm import tqdm
from collections import Counter

def get_most_frequent_sgs(fin, n_classes):
    sg_ids = []
    for line in open(fin, 'r'):
        if len(line) > 1:
            sg_id = line.split("\t")[0]
            sg_id = int(sg_id)
            if sg_id != 0: # 0: non-sg
                sg_ids.append(sg_id)
    sg_id2cnt = Counter(sg_ids)
    sg_ids = [sg_id for sg_id, cnt in sg_id2cnt.most_common(n_classes)]
    idx2sg_id = {idx: sg_id for idx, sg_id in enumerate(sg_ids)}
    sg_id2idx = {sg_id: idx for idx, sg_id in enumerate(sg_ids)}
    return idx2sg_id, sg_id2idx

def prepro(fin, pkl_train, pkl_dev, n_classes, sg_id2idx):
    contexts_li = [[] for _ in range(n_classes)]

    entries = open(fin, 'r').read().split("\n\n")
    for entry in tqdm(entries):
        lines = entry.splitlines()
        for i, line in enumerate(lines):
            if i==0: continue
            cols = line.strip().split("\t")
            sg_id, sent, ids = cols
            sg_id = int(sg_id)
            if sg_id in sg_id2idx:
                idx = sg_id2idx[sg_id]
                ctx = [] # e.g. [ [3, 4, 5], [23, 9, 4, 5]  ]
                for l in lines[:i]:
                    ctx.append([int(id) for id in l.strip().split("\t")[-1].split()])
                contexts = contexts_li[idx]
                contexts.append(ctx)
    train, dev = [], []
    for contexts in contexts_li:
        if len(contexts) > 1:
            train.append(contexts[1:])
            dev.append(contexts[:1])
        else:
            train.append(contexts)
            dev.append([])


    pickle.dump(train, open(pkl_train, 'wb'))
    pickle.dump(dev, open(pkl_dev, 'wb'))
    print("done")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(hp.pkl_train), exist_ok=True)
    os.makedirs(os.path.dirname(hp.pkl_dev), exist_ok=True)

    idx2sg_id, sg_id2idx = get_most_frequent_sgs(hp.text, hp.n_classes)

    phr2sg_id = pickle.load(open(hp.phr2sg_id, 'rb'))
    sg_id2phr = pickle.load(open(hp.sg_id2phr, 'rb'))

    phr2idx = dict()
    for phr, sg_id in phr2sg_id.items():
        if sg_id in sg_id2idx:
            phr2idx[phr] = sg_id2idx[sg_id]

    idx2phr = dict()
    for idx, sg_id in idx2sg_id.items():
        if sg_id in sg_id2phr:
            idx2phr[idx] = sg_id2phr[sg_id]

    pickle.dump(phr2idx, open(hp.phr2idx, 'wb'))
    pickle.dump(idx2phr, open(hp.idx2phr, 'wb'))

    prepro(hp.text, hp.pkl_train, hp.pkl_dev, hp.n_classes, sg_id2idx)
    print("DONE")