import os
import torch
from rouge import Rouge
from datasets import load_dataset
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM)
import time
start = time.time()

def __init__():
    model_ckpt = 'model/Laysumm/PRIMERA-arxiv-TrainPLUStest-4096'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    mode = 'LAYSUMM'        # 'LAYSUMM' or 'ABSTRACT'
    return tokenizer, model, mode

def load_Task2LaysumDataINmydisk():
    tokenizer, model, mode = __init__()
    dataset = load_dataset("scientific_lay_summarisation", "plos")
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

def create_file(path_save):
    with open(path_save, 'w') as f:
        f.write('predict')
        f.write('\n')

def main(model_save_path):
    tokenizer, model, mode = __init__()
    dataset = load_Task2LaysumDataINmydisk()
    create_file(model_save_path)
    # ***** predict *****
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    k = len(dataset['validation'])
    for i in range(k):
        sample_text = dataset['validation'][i]["article"]
        #####? LAYSUMM #####
        if mode == 'LAYSUMM':
            reference = dataset['validation'][i]["summary"]
        #####? ABSTRACT #####
        elif mode == 'ABSTRACT':
            reference = dataset['validation'][i]["abstruct"]
        device = torch.device("cuda")
        model.to('cuda')
        input_ids = tokenizer(sample_text, max_length=4096, truncation=True, padding='max_length', return_tensors='pt').to(device)
        summaries = model.generate(input_ids=input_ids['input_ids'],
                            attention_mask=input_ids['attention_mask'],
                            max_length=256)
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        
        # rouge = Rouge()
        # rouge_score = rouge.get_scores(decoded_summaries[0], reference)
        # rouge_1 += rouge_score[0]["rouge-1"]['f']
        # rouge_2 += rouge_score[0]["rouge-2"]['f']
        # rouge_l += rouge_score[0]["rouge-l"]['f']
        
        with open(model_save_path, 'a') as f:
            # f.write('number' + str(i))
            # f.write('\t')
            # f.write(str(reference))
            # f.write('\t')
            f.write(str(decoded_summaries[0]))
            # f.write('\t')
            # f.write(str(rouge_score[0]["rouge-1"]['f']))
            # f.write('\t')
            # f.write(str(rouge_score[0]["rouge-2"]['f']))
            # f.write('\t')
            # f.write(str(rouge_score[0]["rouge-l"]['f']))
            f.write('\n')
        print('predict number now = ', i)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    path_save = 'out/PRIMERA-arxiv-TrainPLUStest.txt'
    main(path_save)