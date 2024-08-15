import json
import random
import csv

random.seed(13)

strategy_dict = json.load(open("strategies.json", encoding="utf-8"))
strategy_stats = dict()


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
    feedbacks = []
    for idx, turn in enumerate(d["dialog"]):
        utterances.append(_norm(turn["content"]))
        speakers.append(turn["speaker"])
        if "strategy" in turn["annotation"]:
            strategy = turn["annotation"]["strategy"]
            indices.append(idx + 1)
            gold_standards.append(turn["content"])
            strategies.append(strategy_dict[strategy])
            gold_strategies.append(strategy_dict[strategy])
            if strategy not in strategy_stats.keys():
                strategy_stats[strategy] = 1
            else:
                strategy_stats[strategy] += 1
        else:
            strategies.append(-1)
        if "feedback" in turn["annotation"]:
            while len(feedbacks) < len(strategies):
                feedbacks.append(turn["annotation"]["feedback"])
    histories = []
    strategy_histories = []
    speaker_turns = []
    for idx in indices:
        histories.append(" </s> ".join(utterances[max(0, idx - 5): idx]))
        strategy_histories.append(strategies[max(0, idx - 5): idx])
        speaker_turns.append(" ".join(speakers[max(0, idx - 5): idx]))
    dial["samples"] = list(
        zip(histories, strategy_histories, speaker_turns, gold_standards, gold_strategies, feedbacks))
    return dial


def write_split(split_data, split):
    df = []
    for dialog in split_data:
        df.extend(dialog["samples"])
    header = ['Dialogue_History', 'Strategy_History', 'Speaker_Turn', "Gold_Standard", "Next_Strategy", 'Feedback']
    with open(f"{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(df)


raw = json.load(open("ESConv.json", encoding="utf-8"))

data = []
for raw_d in raw:
    data.append(process_data(raw_d))

random.shuffle(data)
dev_size = int(0.15 * len(data))
test_size = int(0.15 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
write_split(train, "train")

print('valid', len(valid))
write_split(valid, "valid")

print('test', len(test))
write_split(test, "test")

total = sum(list(strategy_stats.values()))
print(total)
for k, v in strategy_stats.items():
    strategy_stats[k] = v / total

json.dump(strategy_stats, open("strategy_stats_esconv.json", "w", encoding="utf-8"))
