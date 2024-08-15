import numpy as np
from .spanningtrees.graph import Graph
from .spanningtrees.mst import MST
from .spanningtrees.kbest import KBest


def draw_tree(score_matrix, length):
    score_matrix = -score_matrix  # api finds min trees
    ret = []
    if score_matrix.ndim == 3:
        for sc, l in zip(score_matrix, length):
            sc = sc[:, :l + 1][:l + 1, :]
            sc = np.transpose(sc)
            sc = np.nan_to_num(sc, nan=-float('inf'))
            data = sc.copy()
            sc = Graph.build(sc)
            scs = []
            tree_scores = []
            for i, sc in enumerate(KBest(sc).kbest()):
                cur_tree_score = 0
                for idx, x in enumerate(sc.to_array().tolist()):
                    if idx == 0:
                        continue
                    cur_tree_score += data[x][idx]
                if i == 1:  # find only the best one, i > 1 for future work
                    break

                if len(tree_scores) == 0 or abs(
                        tree_scores[-1] - cur_tree_score) < 0.1:  # the second condition is not triggered
                    scs.append(sc.to_array().tolist())
                tree_scores.append(cur_tree_score)
            # heads = MST(sc).mst().to_array().tolist()
            ret.append(scs)
    else:
        score_matrix = Graph.build(score_matrix)
        heads = next(islice(KBest(sc).kbest(), 1, 2)).to_array().tolist()
        # heads = MST(score_matrix).mst().to_array().tolist()
        ret.append(heads)

    return ret
