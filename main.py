import argparse
import torch
from torch.utils.data import DataLoader
from dataloaders import ESConv, DailyDialogue, ESConvPreProcessed, AnnoMIPreProcessed
from transformers import TrainingArguments
from modules.trainer import TrainerForMulticlassClassification
from modules.roberta import RobertaHeterogeneousGraph
from modules.erc import SequentialERC
from utils import seed_everything

MODELS = {
    "sequential-erc": {
        "model": SequentialERC,
        "trainer": TrainerForMulticlassClassification
    },
    "roberta-hg": {
        "model": RobertaHeterogeneousGraph,
        "trainer": TrainerForMulticlassClassification
    }
}

DATASETS = {
    "esconv": ESConv,
    "esconv-preprocessed": ESConvPreProcessed,
    "dailydialogue": DailyDialogue,
    "annomi-preprocessed": AnnoMIPreProcessed,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--exclude_others', type=int, default=0)
    parser.add_argument('--feedback_threshold', type=int, default=0)
    parser.add_argument('--erc_temperature', type=float, default=0.5)
    parser.add_argument('--erc_mixed', type=int, default=1)
    parser.add_argument('--hg_dim', type=int, default=512)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODELS[args.model]["model"](args)
    model.to(device)
    train_set = DATASETS[args.dataset]("train", args)
    valid_set = DATASETS[args.dataset]("valid", args)
    test_set = DATASETS[args.dataset]("test", args)
    print(f"Total samples: {len(train_set) + len(valid_set) + len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_set.collate_fn)

    training_args = TrainingArguments(
        output_dir=f"./{args.model}-{args.dataset}-checkpoints",
        num_train_epochs=args.total_epochs,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        logging_dir=f'./{args.model}-{args.dataset}-logs',
        learning_rate=args.lr,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    trainer = MODELS[args.model]["trainer"](
        class_weights=torch.tensor(train_set.class_weights),
        model=model,
        deciding_metric="macro f1",
        args=training_args,
        total_steps=args.total_steps,
        id2label=train_set.id2label,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader
    )

    if args.mode == "train":
        trainer.train()
    else:
        trainer.test(args.load_checkpoint)
