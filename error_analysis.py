import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Error Types:
"Information -> Providing Suggestions" 0.37
"Reflection of feelings -> Affirmation and Reassurance" 0.27
"Self-disclosure -> Affirmation and Reassurance" 0.27
"Self-disclosure -> Providing Suggestions" 0.26
"Providing Suggestions -> Affirmation and Reassurance" 0.24
"Information -> Affirmation and Reassurance" 0.23
"Affirmation and Reassurance -> Providing Suggestions" 0.22
"Restatement or Paraphrasing -> Question" 0.22
"Restatement or Paraphrasing -> Affirmation and Reassurance" 0.22
"Reflection of feelings -> Providing Suggestions" 0.21
"""

err_dict = {
    "Information -> Providing Suggestions": [0., 0., 0., 0., 0., 0., 0.],
    "Reflection of feelings -> Affirmation and Reassurance": [0., 0., 0., 0., 0., 0., 0.],
    "Self-disclosure -> Affirmation and Reassurance": [0., 0., 0., 0., 0., 0., 0.],
    "Self-disclosure -> Providing Suggestions": [0., 0., 0., 0., 0., 0., 0.],
    "Providing Suggestions -> Affirmation and Reassurance": [0., 0., 0., 0., 0., 0., 0.],
    "Information -> Affirmation and Reassurance": [0., 0., 0., 0., 0., 0., 0.],
    "Affirmation and Reassurance -> Providing Suggestions": [0., 0., 0., 0., 0., 0., 0.],
    "Restatement or Paraphrasing -> Question": [0., 0., 0., 0., 0., 0., 0.],
    "Restatement or Paraphrasing -> Affirmation and Reassurance": [0., 0., 0., 0., 0., 0., 0.],
    "Reflection of feelings -> Providing Suggestions": [0., 0., 0., 0., 0., 0., 0.],
}

data = []
strategies = [
    "Reflection of feelings",
    "Self-disclosure",
    "Question",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Restatement or Paraphrasing",
    "Information",
    "Others"
]

emotions = [
    'Neutral',
    'Anger',
    'Disgust',
    'Fear',
    'Joy',
    'Sadness',
    'Surprise'
]

emo2id = {emo: _id for _id, emo in enumerate(emotions)}

with open("roberta-hg-esconv-preprocessed-logs/cases.json", "r") as f:
    data = json.load(f)


def get_emo_cat(samp):
    nodes = samp["graph"][0]["nodes"]
    origin_edges = samp["graph"][0]["edges"]
    edges = []
    for edge in origin_edges:
        if edge[1] == 0:
            edges.append(edge)
    attention_weights = samp["attention_weights"]
    edge_weights = []
    if isinstance(attention_weights[0], list):
        for weights in attention_weights:
            edge_weights.append(weights[-1])
    else:
        edge_weights.append(attention_weights[-1])
    edges = list(zip(edges, edge_weights))
    edges = sorted(edges, key=lambda x: x[1], reverse=True)
    emotion_cat = "None"
    for edge in edges:
        node_index = edge[0][0]
        node = nodes[node_index]
        if node in emotions:
            emotion_cat = node
            break
    return emotion_cat

for d in data:
    _id = d["label"].strip() + " -> " + d["prediction"].strip()
    if _id in err_dict:
        emo_cat = get_emo_cat(d)
        if emo_cat != "None":
            err_dict[_id][emo2id[emo_cat]] += 1.

for tp, dis in err_dict.items():
    err_dict[tp] = [i / sum(dis) for i in dis]

err_types = [
    "Information -> Providing Suggestions",
    "Reflection of feelings -> Affirmation and Reassurance",
    "Self-disclosure -> Affirmation and Reassurance",
    "Self-disclosure -> Providing Suggestions",
    "Providing Suggestions -> Affirmation and Reassurance",
    "Information -> Affirmation and Reassurance",
    "Affirmation and Reassurance -> Providing Suggestions",
    "Restatement or Paraphrasing -> Question",
    "Restatement or Paraphrasing -> Affirmation and Reassurance",
    "Reflection of feelings -> Providing Suggestions",
]

heatmap = []

for i in range(len(err_types)):
    heatmap.append(err_dict[err_types[i]])

fig, ax = plt.subplots(figsize=(8.0, 3.8))

sns.heatmap(heatmap, annot=True, cmap='Blues', vmin=0, vmax=1, ax=ax,
            yticklabels=[], cbar=False, fmt='.2f')  # cbar=False 移除colorbar

ax.set_xticklabels(emotions, rotation=45, ha='right')

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
for i, label in enumerate(err_types):
    ax.text(len(emotions) + 0.1, i + 0.5, label,
            ha='left', va='center', rotation=0)

plt.tight_layout()

plt.savefig('emotion_error_heatmap.pdf', format='pdf', bbox_inches='tight')

plt.savefig('emotion_error_heatmap.svg', format='svg', bbox_inches='tight')

plt.show()
