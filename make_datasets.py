import csv
import pandas as pd
import os
import json

DATASETS = ["esconv", "annomi"]


for dataset in DATASETS:
    source_path = os.path.join("data", dataset)
    output_path = os.path.join("data", dataset + "_datasets")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(os.path.join(source_path, "strategies.json"), "r") as f:
        strategy2id = json.load(f)
    id2strategy = {v: k for k, v in strategy2id.items()}
    for split in ["train", "valid", "test"]:
        source_file = os.path.join(source_path, split + ".csv")
        tf = pd.read_csv(source_file)
        text = []
        label = []
        for index, row in tf.iterrows():
            label.append(row["Next_Strategy"])
            # flatten context
            context = row["Dialogue_History"]
            speaker_turn = str(row["Speaker_Turn"])
            speakers = speaker_turn.split(" ")
            strategy_history = row["Strategy_History"][1:-1]
            strategies = [int(s) for s in strategy_history.split(", ")]
            utterances = context.split("</s>")
            flattened_context = ""
            assert len(speakers) == len(utterances)
            assert len(speakers) == len(strategies)
            for i, turn in enumerate(utterances):
                new_turn = ""
                # check speaker
                if speakers[i] == "seeker":
                    new_turn += "[seeker]"
                elif speakers[i] == "supporter":
                    new_turn += f"[supporter] [{id2strategy[strategies[i]]}] "
                new_turn += turn
                flattened_context += new_turn
            text.append(flattened_context)
        with open(os.path.join(output_path, split + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "text"])
            for i in range(len(text)):
                writer.writerow([label[i], text[i]])
