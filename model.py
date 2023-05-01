import os
import numpy as np
import torch
from rouge import Rouge
from datasets import load_dataset
from datasets import load_metric
from datasets import concatenate_datasets
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM)
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import time

def __init__():
    mode = 'ABSTRACT'        # 'LAYSUMM' or 'ABSTRACT'
    model_ckpt = "allenai/PRIMERA"
    model_save_path = 'model/Abstract/PRIMERA-TrainPLUStest-4096'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    training_args = TrainingArguments(
                                output_dir='model',
                                num_train_epochs=10,
                                warmup_steps=500,
                                per_device_train_batch_size=2,
                                per_device_eval_batch_size=2,
                                weight_decay=0.01,
                                logging_steps=10,
                                push_to_hub=False, 
                                evaluation_strategy='steps',
                                eval_steps=1000,
                                save_steps=1e6,
                                gradient_accumulation_steps=16,
                                fp16 = True,
                                learning_rate=3e-5,
                                )
    return tokenizer, model, training_args, mode, model_save_path

def save_Task2LaysumData2mydisk():
    print('***** see [ def save_Task2LaysumData2mydisk() ] *****')
    """
    input in terminal
    ```sh=
    git lfs install
    git clone https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation
    ```
    """
    return 0

def load_Task2LaysumDataINmydisk():
    tokenizer, model, training_args, mode, model_save_path = __init__()
    dataset = load_dataset("/workplace/jhyang/BioNLP2023/_codalab_train/scientific_lay_summarisation", "plos")
    #####? LAYSUMM #####
    if mode == 'LAYSUMM':
        dataset = dataset.map(lambda example: {'article': ' <doc-sep> '.join(example['article'].split('\n')[1:])})
    #####? ABSTRACT #####
    elif mode == 'ABSTRACT':
        dataset = dataset.map(lambda example: {'abstruct': ' '.join(example['article'].split('\n')[:1])}, remove_columns=['year'])
        dataset = dataset.map(lambda example: {'article': ' <doc-sep> '.join(example['article'].split('\n')[1:])})
    else: 
        return 0
    return dataset

def convert_examples_to_features(example_batch):
    tokenizer, model, training_args, mode, model_save_path = __init__()
    input_encodings = tokenizer(example_batch["article"], max_length=4096, truncation=True)
    #####? LAYSUMM #####
    if mode == 'LAYSUMM':
        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(example_batch["summary"], max_length=256, truncation=True)
    #####? ABSTRACT #####
    elif mode == 'ABSTRACT':
        with tokenizer.as_target_tokenizer():
                target_encodings = tokenizer(example_batch["abstruct"], max_length=256, truncation=True)
    else: 
        return 0    
    return {"input_ids": input_encodings["input_ids"], 
           "attention_mask": input_encodings["attention_mask"], 
           "labels": target_encodings["input_ids"]}

def compute_metrics(eval_pred):
    tokenizer, model, training_args, model_save_path = __init__()
    rouge_metric = load_metric("rouge")
    predictions, labels = eval_pred
    # 把 DataCollatorForSeq2Seq 會填入的 -100 排除掉
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
    
def main():
    start = time.time()
    tokenizer, model, training_args, mode, model_save_path = __init__()
    dataset = load_Task2LaysumDataINmydisk()
    dataset_pt = dataset.map(convert_examples_to_features, batched=True)
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    print(concatenate_datasets([dataset_pt['train'], dataset_pt['test']]))
    print(dataset_pt["validation"])
    
    ### Train ###
    trainer = Trainer(model=model,
                  args=training_args,
                  tokenizer=tokenizer,
                  data_collator=seq2seq_data_collator,
                  train_dataset=concatenate_datasets([dataset_pt['train'], dataset_pt['test']]), 
                  eval_dataset=dataset_pt["validation"],
                #   compute_metrics=compute_metrics,
                  )
    trainer.train()
    trainer.save_model(model_save_path)
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    print('model_save = ', model_save_path)
    
    eval_results = trainer.evaluate(test_dataset=dataset_pt["test"])
    print(eval_results)

if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    
    if not os.path.exists('scientific_lay_summarisation'):
        print('***** see [ def save_Task2LaysumData2mydisk() ] *****')
    else:
        print()
        print('***** Train Now *****')
        print()
        main()
