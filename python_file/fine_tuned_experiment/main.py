from datasets import load_dataset
from overlap_tokenizer import tokenizer
from model import overlap_model
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import os

from transformers import TrainerCallback

METRIC = evaluate.load("f1")

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_filepath = os.path.join(args.output_dir, "experiment_metrics.log")
            with open(log_filepath, "a") as f:
                f.write(str(logs) + "\n")
                f.flush()
                os.fsync(f.fileno())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return METRIC.compute(predictions=predictions, references=labels, average="macro")


def main():
    print("--- SCRIPT STARTING ---", flush=True)
    import os
    print(f"Current Working Directory: {os.getcwd()}", flush=True)

    model_name = "microsoft/deberta-v3-base"
    overlap_tokenizer = tokenizer(model_name)
    model = overlap_model(model_name)

    model.resize_embeddings(len(overlap_tokenizer.tokenizer))

    dataset = load_dataset('json', data_files={'train': "../../data/FINAL_DATA_TO_RUN/data_with_edges.json"})

    split_dataset = dataset["train"].train_test_split(test_size=0.2)

    tokenized_dataset = split_dataset.map(
        lambda x: overlap_tokenizer.tokenize_function(x, model.label2id),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir="./overlap_output",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    os.makedirs("./overlap_output", exist_ok=True)
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=overlap_tokenizer.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()]
    )

    trainer.train()