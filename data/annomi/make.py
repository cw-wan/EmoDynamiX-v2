import json
import random
import csv

random.seed(77)

strategy_dict = json.load(open("strategies.json", encoding="utf-8"))


def _norm(x):
    return ' '.join(x.strip().split())


def process_data(d):
    dial = {}
    utterances = ["<START>", ]
    speakers = ["None", ]
    strategies = [-1, ]
    indices = []
    gold_standards = []
    gold_strategies = []
    for idx, turn in enumerate(d["dialog"]):
        utterances.append(_norm(turn["content"]))
        speaker = "seeker" if turn["interlocutor"] == "client" else "supporter"
        speakers.append(speaker)
        if turn["annotation"]:
            indices.append(idx + 1)
            gold_standards.append(turn["content"])
            strategies.append(strategy_dict[turn["annotation"]])
            gold_strategies.append(strategy_dict[turn["annotation"]])
        else:
            strategies.append(-1)
    histories = []
    strategy_histories = []
    speaker_turns = []
    for idx in indices:
        histories.append(" </s> ".join(utterances[max(0, idx - 5): idx]))
        strategy_histories.append(strategies[max(0, idx - 5): idx])
        speaker_turns.append(" ".join(speakers[max(0, idx - 5): idx]))
    dial["samples"] = list(
        zip(histories, strategy_histories, speaker_turns, gold_standards, gold_strategies))
    return dial


def write_split(split_data, split):
    df = []
    for dialog in split_data:
        df.extend(dialog["samples"])
    header = ['Dialogue_History', 'Strategy_History', 'Speaker_Turn', "Gold_Standard", "Next_Strategy"]
    with open(f"{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(df)


raw = json.load(open("AnnoMI-clean.json", encoding="utf-8"))
filtered = []
for d in raw:
    if d["quality"] == "high":
        filtered.append(d)
raw = filtered

data = []
for raw_d in raw:
    data.append(process_data(raw_d))

random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
test = data[:test_size]
valid = data[test_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
write_split(train, "train")

print('valid', len(valid))
write_split(valid, "valid")

print('test', len(test))
write_split(test, "test")
