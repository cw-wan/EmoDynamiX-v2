import torch
import os
import json
import datetime
from utils import write_log
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import transformers
from transformers import AdamW, TrainingArguments
from metrics import preference_bias


class TrainerForMulticlassClassification:
    def __init__(self,
                 args: TrainingArguments = None,
                 total_steps: int = None,
                 deciding_metric: str = None,
                 class_weights: torch.Tensor = None,
                 id2label: dict = None,
                 model: nn.Module = None,
                 train_loader: DataLoader = None,
                 valid_loader: DataLoader = None,
                 test_loader: DataLoader = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.total_steps = total_steps
        self.deciding_metric = deciding_metric
        self.model = model
        self.train_loader = train_loader
        self.id2label = id2label
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.best_ckpt = None

        if not os.path.exists(self.args.logging_dir):
            os.makedirs(self.args.logging_dir)

    def evaluate(self, loader, case_study=False):
        cases = []
        self.model.eval()
        with torch.no_grad():
            predictions = []
            truths = []
            bar = tqdm(loader)
            for _, batch in enumerate(bar):
                outputs = self.model(batch)
                y_pred = outputs.get("logits")
                if case_study:
                    contexts = batch["dialogue_history"]
                    preds = torch.argmax(y_pred, dim=-1).int().cpu().detach().numpy()
                    labels = batch.get("label").detach().numpy()
                    for i in range(len(contexts)):
                        cases.append({
                            "dialogue_history": [contexts[i], ],
                            "strategy_history": [batch["strategy_history"][i], ],
                            "speaker_turn": [str(batch["speaker_turn"][i]), ],
                            "prediction": self.id2label[preds[i]],
                            "label": self.id2label[labels[i]]
                        })
                predictions.append(torch.argmax(y_pred, dim=-1))
                truths.append(batch.get("label"))
            predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
            truths = torch.cat(truths, dim=-1).detach().numpy()
            acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average=None)
            weighted_f1 = f1_score(truths, predictions, average='weighted')
            macro_f1 = f1_score(truths, predictions, average='macro')
            micro_f1 = f1_score(truths, predictions, average='micro')
            c_matrix = confusion_matrix(truths, predictions)
            metrics = {
                'accuracy': acc,
                'macro f1': macro_f1,
                'micro f1': micro_f1,
                'weighted f1': weighted_f1,
                'confusion matrix': c_matrix.tolist(),
                'preference bias': preference_bias(c_matrix)
            }
            for _id in range(len(self.id2label.keys())):
                metrics[self.id2label[_id]] = f1[_id]
            if case_study:
                return metrics, cases
            return metrics

    def train(self):
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime('%Y-%m-%d-%H')
        log_path = os.path.join(self.args.logging_dir, f'train{datetime_string}.log')

        best_checkpoint = 0
        best_metric = 0

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_loader)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=self.args.warmup_steps,
                                                                              num_training_steps=int(total_steps))
        step_counter = 0
        for epoch in range(1, int(self.args.num_train_epochs) + 1):
            stop_training = 0
            self.model.train()
            total_loss = 0
            loss = torch.tensor(0)
            bar = tqdm(self.train_loader)
            for idx, batch in enumerate(bar):
                optimizer.zero_grad()
                step_counter += 1
                bar.set_description(f"Epoch {epoch}| Step {step_counter} | Loss: {loss:.4f}")
                outputs = self.model(batch)
                loss = self.cross_entropy_loss(outputs.get("logits"), batch.get("label").to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                if step_counter % self.args.save_steps == 0:
                    if not os.path.exists(self.args.output_dir):
                        os.makedirs(self.args.output_dir)
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{step_counter}.pth")
                    self.model.save(save_path)
                if step_counter % self.args.eval_steps == 0:
                    metrics = self.evaluate(self.valid_loader)
                    if metrics[self.deciding_metric] > best_metric:
                        best_metric = metrics[self.deciding_metric]
                        best_checkpoint = step_counter
                    msg = f"Evaluation Step {step_counter} | "
                    for k, v in metrics.items():
                        if k != "confusion matrix":
                            msg += f"{k}: {v:.4f}, "
                    print(msg)
                    write_log(msg, log_path)
                if step_counter == self.total_steps:
                    stop_training = 1
                    break
            if stop_training:
                break
        print(f"Best checkpoint: Step {best_checkpoint}")
        self.best_ckpt = best_checkpoint
        self.test()

    def test(self, ckpt=None):
        load_ckpt = ckpt if ckpt else self.best_ckpt
        print("Testing ...")
        ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-{load_ckpt}.pth")
        self.model.load(ckpt_path)
        test_metrics, cases = self.evaluate(self.test_loader, case_study=True)
        report_path = os.path.join(self.args.logging_dir, f'result.json')
        json.dump(test_metrics, open(report_path, "w", encoding="utf-8"))
        cases_path = os.path.join(self.args.logging_dir, f'cases.json')
        json.dump(cases, open(cases_path, "w", encoding="utf-8"))
        msg = f"Test result for checkpoint {load_ckpt} | "
        for k, v in test_metrics.items():
            if k != "confusion matrix":
                msg += f"{k}: {v:.4f}, "
        print(msg)
