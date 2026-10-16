import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    BertModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from model import MAX_LENGTH, NLI, NLIConfig, collate_fn


BASE_MODEL = "microsoft/MiniLM-L12-H384-uncased"
MODEL_DIR = "./MODEL"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = load_dataset("nyu-mll/multi_nli")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # Keep only the three labels required by the assignment.
    raw_train = dataset["train"].filter(
        lambda example: example["label"] in (0, 1, 2)
    )
    raw_validation = dataset["validation_matched"].filter(
        lambda example: example["label"] in (0, 1, 2)
    )

    def preprocess(batch):
        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        encoded["labels"] = batch["label"]
        return encoded

    train_data = raw_train.map(
        preprocess,
        batched=True,
        remove_columns=raw_train.column_names,
        desc="Preparing MNLI train",
    )
    validation_data = raw_validation.map(
        preprocess,
        batched=True,
        remove_columns=raw_validation.column_names,
        desc="Preparing MNLI validation",
    )

    # Initialize the custom NLI class with pretrained MiniLM weights. The
    # submitted MODEL directory will contain all weights needed by the grader.
    pretrained_encoder = BertModel.from_pretrained(BASE_MODEL)
    base_config = pretrained_encoder.config
    config = NLIConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        nclass=3,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        intermediate_size=base_config.intermediate_size,
        max_position_embeddings=base_config.max_position_embeddings,
        type_vocab_size=base_config.type_vocab_size,
        hidden_dropout_prob=base_config.hidden_dropout_prob,
        attention_probs_dropout_prob=base_config.attention_probs_dropout_prob,
        pad_token_id=tokenizer.pad_token_id,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    model = NLI(config)
    model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
    del pretrained_encoder

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    if parameter_count > 40_000_000:
        raise ValueError("Model exceeds the 40-million-parameter limit")

    training_args = TrainingArguments(
        output_dir="./NLIMODEL",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        logging_steps=200,
        report_to="none",
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Best model and tokenizer saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
