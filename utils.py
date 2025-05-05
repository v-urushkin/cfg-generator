from typing import List


def split_by_punct(x: List[str]) -> List[List[str]]:
    puncts = ('.', '?', '!')
    res = []
    left = 0
    for right, rsym in enumerate(x):
        if rsym in puncts:
            res.append(x[left:right+1])
            left = right+1
    if len(res) == 0:
        res.append(x)
    return res
