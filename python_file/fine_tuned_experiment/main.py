from datasets import load_dataset
from overlap_tokenizer import tokenizer
from model import overlap_model
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

METRIC = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return METRIC.compute(predictions=predictions, references=labels, average="macro")


def main():
    model_name = "microsoft/deberta-v3-base"
    overlap_tokenizer = tokenizer(model_name)
    model = overlap_model(model_name)

    model.resize_embeddings(len(overlap_tokenizer.tokenizer))

    dataset = load_dataset('json', data_files={'train': "../../data/FINAL_DATA_TO_RUN/data_with_edges.json"})

    tokenized_dataset = dataset.map(
        lambda x: overlap_tokenizer.tokenize_function(x, model.label2id),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir="./overlap_output",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=overlap_tokenizer.tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()