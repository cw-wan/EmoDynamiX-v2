import random
import json
import os

random.seed(77)

strategy_dict = json.load(open("strategies.json", encoding="utf-8"))
id2strategy = {v: k for k, v in strategy_dict.items()}

if not os.path.exists('blenderbot-joint'):
    os.mkdir('blenderbot-joint')


def _norm(x):
    return ' '.join(x.strip().split())


def process_data(d):
    sample = {
        "emotion_type": "",
        "problem_type": d["topic"],
        "situation": "",
        "dialog": []
    }
    for turn in d["dialog"]:
        u = {
            "text": turn["content"],
            "speaker": "usr" if turn["interlocutor"] == "client" else "sys"
        }
        if turn["interlocutor"] == "therapist":
            u["strategy"] = turn["annotation"]
        sample["dialog"].append(u)
    return sample


def write_split(split_data, split):
    output_path = os.path.join("blenderbot-joint/", f"{split}.txt")
    with open(output_path, "w+", encoding="utf-8") as f:
        for sample in split_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


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

for k, v in id2strategy.items():
    id2strategy[k] = f"[{v}]"

with open(os.path.join("blenderbot-joint/", "strategy.json"), "w", encoding="utf-8") as f:
    json.dump(id2strategy, f)
