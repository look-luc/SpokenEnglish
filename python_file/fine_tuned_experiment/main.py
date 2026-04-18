import transformers
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

    f1 = METRIC.compute(predictions=predictions, references=labels, average="macro")["f1"]
    acc = evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"]

    return {
        "f1": f1,
        "accuracy": acc
    }


def main():
    print("--- SCRIPT STARTING ---", flush=True)
    import os
    print(f"Current Working Directory: {os.getcwd()}", flush=True)

    model_name = "YituTech/conv-bert-base"
    overlap_tokenizer = tokenizer(model_name)
    model = overlap_model(model_name)

    model.resize_embeddings(len(overlap_tokenizer.tokenizer))

    dataset = load_dataset('json', data_files={'train': "../../data/FINAL_DATA_TO_RUN/data_with_edges.json"})

    split_dataset = dataset["train"].train_test_split(test_size=0.2)

    tokenized_dataset = split_dataset.map(
        lambda x: overlap_tokenizer.tokenize_function(x, model.label2id),
        batched=True
    )

    # In main.py
    training_args = TrainingArguments(
        output_dir="./overlap_output",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.15,
        weight_decay=0.01,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=False,
        fp16=False,
        report_to="none",
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

    print("\n--- FINAL EVALUATION ---")
    final_metrics = trainer.evaluate()

    print(f"Overall F1 Score: {final_metrics.get('eval_f1'):.4f}")
    print(f"Overall Accuracy: {final_metrics.get('eval_accuracy'):.4f}")

if __name__ == "__main__":
    main()
    print("--- SCRIPT  ENDING ---", flush=True)