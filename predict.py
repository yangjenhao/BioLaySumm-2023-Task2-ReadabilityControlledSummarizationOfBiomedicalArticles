import torch
from rouge import Rouge
from datasets import load_dataset
import pandas as pd
import json
from datasets import Dataset
import time
start = time.time()
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModel
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM,LEDConfig,LEDForConditionalGeneration,BartForConditionalGeneration)

def __init__():
    model_ckpt = 'model/Laysumm/PRIMERA'
    mode = 'PRIMERA'        # 'PRIMERA' or 'None'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    return model_ckpt, tokenizer, model, mode

def load_jsonl2data(path):
    model_ckpt, tokenizer, model, mode = __init__()
    # 讀取JSONL檔案中的所有行
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 將每一行JSON轉換成字典
    data = [json.loads(line) for line in lines]
    # 將字典列表轉換成DataFrame
    df = pd.DataFrame(data)
    df_dict = df.to_dict("list")
    dataset = Dataset.from_dict(df_dict)
    # dataset.map處理資料
    if mode == 'None':
        dataset = dataset.map(lambda example: {'article': ' '.join(example['article'].split('\n'))})
    elif mode == 'PRIMERA':
        dataset = dataset.map(lambda example: {'article': ' <doc-sep> '.join(example['article'].split('\n'))})
    else: 
        return 0
    return dataset, len(dataset)

def create_txt_file(model_ckpt, save_file_name):
    with open(save_file_name, 'w') as f:
        f.write(str(model_ckpt))
        f.write('\t')
        f.write('predict')
        f.write('\n')
def main(test_file, save_file_name):
    model_ckpt, tokenizer, model, mode = __init__()
    create_txt_file(model_ckpt, save_file_name)
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    dataset, len_dataset = load_jsonl2data(test_file)
    
    ### predict ###
    k = len_dataset
    for i in range(k):
        sample_text = dataset[i]["article"]
        device = torch.device("cuda")
        model.to('cuda')
        input_ids = tokenizer(sample_text, max_length=4096, truncation=True, padding='max_length', return_tensors='pt').to(device)
        summaries = model.generate(input_ids=input_ids['input_ids'],
                                    attention_mask=input_ids['attention_mask'],
                                    max_length=256)
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]

        
        with open(save_file_name, 'a') as f:
            f.write(str(decoded_summaries[0]))
            f.write('\n')
        print('predict number now = ', i)



if __name__ == '__main__':
    test_file = 'data/test.jsonl'
    save_file_name = 'laysumm.txt' # named `abstract.txt` and `laysumm.txt`
    main(test_file, save_file_name)
    