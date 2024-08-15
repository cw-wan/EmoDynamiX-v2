import sys
import torch
import os
from modules.sddp import StructuredDialogueDiscourseParser
from modules.erc import SequentialERC
import pandas as pd
from tqdm import tqdm
import pickle


def main(dataset):
    save_path = f"data/{dataset}_preprocessed"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dialogue_parser = StructuredDialogueDiscourseParser(ckpt_path="pre_trained_models/sddp_stac")
    erc = SequentialERC()
    erc.load("pre_trained_models/sequential_erc_model.pth")
    erc.to(device)
    erc.eval()

    def preprocess(split):
        print("Preprocessing {} ...".format(split))
        tf = pd.read_csv(f'data/{dataset}/{split}.csv')
        data = []
        inputs = []
        for i, row in tf.iterrows():
            utterances = row["Dialogue_History"].split("</s>")
            speakers = str(row["Speaker_Turn"]).split(" ")
            dialogue_for_parsing = []
            erc_context = ""
            for j in range(len(utterances)):
                turn = {
                    "speaker": speakers[j],
                    "text": utterances[j]
                }
                dialogue_for_parsing.append(turn)
                erc_context += " </s> " + utterances[j]
            dialogue_parser_input = [dialogue_for_parsing, ]
            erc_input = {
                "texts": [erc_context, ]
            }
            d = {
                "dialogue_history": row["Dialogue_History"],
                "strategy_history": row["Strategy_History"],
                "speaker_turn": row["Speaker_Turn"],
                "gold_standard": row["Gold_Standard"],
                "label": row["Next_Strategy"],
            }
            data.append(d)
            inputs.append({
                "parser_input": dialogue_parser_input,
                "erc_input": erc_input,
            })
        bar = tqdm(inputs)
        for idx, input in enumerate(bar):
            parsed_dialogue = dialogue_parser.parse(input["parser_input"])[0]
            with torch.no_grad():
                erc_logits = erc(input["erc_input"])["logits"]
            data[idx]["parsed_dialogue"] = parsed_dialogue
            data[idx]["erc_logits"] = erc_logits.detach().cpu().numpy()
        filtered = []
        for d in data:
            if d["erc_logits"].shape[0] == len(str(d["speaker_turn"]).split(" ")):
                filtered.append(d)
        data = filtered
        output_path = os.path.join(save_path, f"{split}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    for split in ["train", "valid", "test"]:
        preprocess(split)


if __name__ == "__main__":
    main(sys.argv[1])
