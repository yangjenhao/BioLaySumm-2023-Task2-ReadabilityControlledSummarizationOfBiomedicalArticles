import textstat
from rouge import Rouge
from evaluate import load

def Average(lst):
    return sum(lst) / len(lst)

def Rouge_all(predict, src):
    src, target = src, predict
    rouge = Rouge()
    rouge_score = rouge.get_scores(src, target)
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    for i in range(len(rouge_score)):
        rouge_1 += rouge_score[i]["rouge-1"]['f']
        rouge_2 += rouge_score[i]["rouge-2"]['f']
        rouge_l += rouge_score[i]["rouge-l"]['f']
        
    rouge_1 = rouge_1 / len(rouge_score)
    rouge_2 = rouge_2 / len(rouge_score)
    rouge_l = rouge_l / len(rouge_score)
    print('rouge-1 :', round(rouge_1, 4))
    print('rouge-2 :', round(rouge_2, 4))
    print('rouge-L :', round(rouge_l, 4))
    
def FKGL_DCRS_score(predict,summary):
    FKGL = []
    for i in predict:
        FKGL.append(textstat.flesch_kincaid_grade(i))

    print(f'FKGL : {round(Average(FKGL)/100, 4)}')

    DCRS = []
    for i in predict:
        DCRS.append(textstat.dale_chall_readability_score(i))
    print(f'DCRS : {round(Average(DCRS)/100, 4)}')
    
def bertscore(predict,summary):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predict, references=summary, model_type="distilbert-base-uncased")
    print("BERTscore precision: ",Average(results["precision"]))
    print("BERTscore recall: ",Average(results["recall"]))
    print("BERTscore f1: ",Average(results["f1"]))
    
if __name__ == '__main__':
    """
    score_file : `out/FileExaple.txt` an exaple for input.
    """
    score_file = 'out/PRIMER.txt'
    src, predict = [], []
    with open(score_file) as f:
        Flag = False
        for line in f.readlines():
            if Flag == True:
                line = line.split('\t')
                src.append(line[1])
                predict.append(line[2])
            Flag = True
            
    Rouge_all(predict, src)
    FKGL_DCRS_score(predict, src)
    bertscore(predict, src)