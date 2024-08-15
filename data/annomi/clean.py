import pandas as pd
import json
import matplotlib.pyplot as plt

merge = True

dialogues = dict()
strategy_stats = dict()
total_strategies = 0

df = pd.read_csv("AnnoMI-full.csv")
for _, row in df.iterrows():
    if row["transcript_id"] not in dialogues:
        dialogues[row["transcript_id"]] = {
            "dialogue_id": row["transcript_id"],
            "quality": row["mi_quality"],
            "video_title": row["video_title"],
            "video_url": row["video_url"],
            "topic": row["topic"],
            "dialog": dict()
        }
    if row["utterance_id"] not in dialogues[row["transcript_id"]]["dialog"]:
        utterance = {
            "utterance_id": row["utterance_id"],
            "interlocutor": row["interlocutor"],
            "content": row["utterance_text"],
            "annotation": dict()
        }
        dialogues[row["transcript_id"]]["dialog"][row["utterance_id"]] = utterance
    if row["interlocutor"] == "therapist":
        annotation = "Other"
        if row["therapist_input_exists"]:
            input_type = row["therapist_input_subtype"]
            if merge:
                if input_type == "options" or input_type == "advice" or input_type == "negotiation":
                    input_type = "suggestion"
            annotation = f"Provide {input_type}"
        elif row["reflection_exists"]:
            annotation = f"Reflection {row['reflection_subtype']}"
        elif row["question_exists"]:
            annotation = f"Question {row['question_subtype']}"
        if annotation not in strategy_stats.keys():
            strategy_stats[annotation] = 1
        else:
            strategy_stats[annotation] += 1
        total_strategies += 1
        if annotation not in dialogues[row["transcript_id"]]["dialog"][row["utterance_id"]]["annotation"]:
            dialogues[row["transcript_id"]]["dialog"][row["utterance_id"]]["annotation"][annotation] = 1
        else:
            dialogues[row["transcript_id"]]["dialog"][row["utterance_id"]]["annotation"][annotation] += 1

dialogue_list = []
for _, d in dialogues.items():
    u_list = []
    for _, u in d["dialog"].items():
        if u["interlocutor"] == "therapist":
            st = list(zip(u["annotation"].keys(), u["annotation"].values()))
            st_ranking = sorted(st, key=lambda x: x[1], reverse=True)
            annotation = st_ranking[0][0]
            u["annotation"] = annotation
        else:
            u["annotation"] = None
        u_list.append(u)
    d["dialog"] = u_list
    dialogue_list.append(d)

json.dump(dialogue_list, open("AnnoMI-clean.json", "w"))
for s, cnt in strategy_stats.items():
    strategy_stats[s] = cnt / total_strategies

json.dump(strategy_stats, open("strategy_stats_annomi.json", "w"))
