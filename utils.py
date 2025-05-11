from typing import List, Set, Optional


def preprocess(x: List[str], NNP: Set[Optional[str]] = set()) -> List[List[str]]:
    puncts = ('.', '?', '!')
    res = []
    subseq = []
    for sym in x:
        tsym = sym[:]
        if sym not in NNP:
            tsym = sym.lower()
        subseq.append(tsym)
        if sym in puncts:
            if subseq[0] in (',', ':', ';'):  # if the first symbol is comma than drop it
                subseq = subseq[1:]
            subseq[0] = subseq[0].capitalize()
            res.append(subseq)
            subseq = []
    if len(res) == 0 or len(subseq) != 0:
        if subseq[0] in (',', ':', ';'):
                subseq = subseq[1:]
        res.append(subseq)
    return res
