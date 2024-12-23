#Copyright (2024) Bytedance Ltd. 
import os
import argparse
import json
import numpy as np

def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['answer'], str):
            elem['answer'] = [elem['answer']]
        score = max([
            (1.0 if
            (ann.strip().lower().replace(".","") in  elem['predict'].strip().lower().replace(".","") ) else 0.0)
            for ann in elem['answer']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

    
parser = argparse.ArgumentParser()
parser.add_argument("--model_result_path", default='evaluation/sample_result')
args = parser.parse_args()
acc_items = []
for line in sorted(os.listdir(args.model_result_path)):
    items = json.load(open(os.path.join(args.model_result_path,line)))
    lang_acc = evaluate_exact_match_accuracy(items)
    acc_items.append(lang_acc)
    print(line.strip('.json'), 'acc:', lang_acc)
print('Average acc:', np.mean(acc_items))