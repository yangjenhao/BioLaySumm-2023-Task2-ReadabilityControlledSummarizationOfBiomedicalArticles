# BioLaySumm-2023-Task2-Readability-Controlled-Summarization-of-BiomedicalArticles

Link: https://biolaysumm.org/

Team: NCUEE

# Introduction from BioLaySumm 2023

Biomedical publications contain the latest research on prominent health-related topics, ranging from common illnesses to global pandemics. This can often result in their content being of interest to a wide variety of audiences including researchers, medical professionals, journalists, and even members of the public. However, the highly technical and specialist language used within such articles typically make it difficult for non-expert audiences to understand their contents.

Abstractive summarization models can be used to generate a concise summary of an article, capturing its salient point using words and sentences that aren’t used in the original text. As such, these models have the potential to help broaden access to highly technical documents when trained to generate summaries that are more readable, containing more background information and less technical terminology (i.e., a “lay summary”).

This shared task surrounds the abstractive summarization of biomedical articles, with an emphasis on controllability and catering to non-expert audiences. Through this task, we aim to help foster increased research interest in controllable summarization that helps broaden access to technical texts and progress toward more usable abstractive summarization models in the biomedical domain.

# Subtask 2 Results

### Rank 

| Criterion  | 1st                                   | 2nd        | 3rd                  |
| ---------- | ------------------------------------- | ---------- | -------------------- |
| Relevance  | NCUEE-NLP                             | LHS712EE   | Pathology Dynamics   |
| Readability| Pathology Dynamics                   | NCUEE-NLP  | LHS712EE             |
| Factuality | Baseline                              | LHS712EE   | Pathology Dynamics   |
| Overall    | Pathology Dynamics, NCUEE-NLP, LHS712EE| -          | -                    |


### Score Detail

| # | Team          | rouge_1 | rouge_2 | rouge_L | bertscore | FKGL   | DCRS   | BARTScore |
|---|---------------|---------|---------|---------|-----------|--------|--------|-----------|
| 1 | NCUEE-NLP     | 0.4514  | 0.1402  | 0.4123  | 0.8545    | 2.0475 | 0.9340 | -2.1102   |
| 2 | LHS712EE      | 0.4417  | 0.1299  | 0.4053  | 0.8549    | 2.2634 | 0.9364 | -1.1403   |
| 3 | TGoldsack     | 0.4088  | 0.1163  | 0.3686  | 0.8549    | 2.3961 | 0.9312 | -0.9783   |
| 4 | Pathology Dynamics | 0.4511  | 0.1382  | 0.4100  | 0.8532    | 2.1067 | 0.8232 | -1.5682   |


# Usage:

Step 1.  
```
pip install -r requirements.txt
```

Step 2.

* `eval.py`: This file is used for evaluation purposes.

* `model.py`: This file typically contains code for defining and training a machine learning model.

* `predict.py`: This file contains code to use a trained model to make predictions on new data.

* `score.py`: This file is used to calculate a score or metric for a set of predictions made by a model.

# Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.