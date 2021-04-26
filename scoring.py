# from statistics import mean
from bert_score import score
# from rouge_score import rouge_scorer
from datasets import load_metric

# Load data 
print("Load data")
ref = []
with open('ref.txt', 'r') as f:
  for line in f.readlines():
    ref.append(line.strip())

out = []
with open('out.txt', 'r') as f:
  for line in f.readlines():
    out.append(line.strip())

print("Calculate ROUGE")
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
# total_scores = {'r1':[], 'r2':[], 'rl':[], 'rlsum':[]}

# for i in range(len(ref)):
#     scores = scorer.score(ref[i], out[i])
#     r1 = scores['rouge1'][2]
#     r2 = scores['rouge2'][2]
#     rl = scores['rougeL'][2]
#     rlsum = scores['rougeLsum'][2]

#     total_scores['r1'].append(r1)
#     total_scores['r2'].append(r2)
#     total_scores['rl'].append(rl)
#     total_scores['rlsum'].append(rlsum)

# r1=mean(total_scores['r1'])
# r2=mean(total_scores['r2'])
# rl=mean(total_scores['rl'])
# rlsum=mean(total_scores['rlsum'])

metric = load_metric("rouge")
rouge = metric.compute(predictions=out, references=ref)
r1 = rouge['rouge1'].mid
r2 = rouge['rouge2'].mid
rl = rouge['rougeL'].mid
rlsum = rouge['rougeLsum'].mid

print("ROUGE F1")
print("R1: ", r1.fmeasure)
print("R2: ", r2.fmeasure)
print("RL: ", rl.fmeasure)
print("RLsum: ", rlsum.fmeasure)

print()

P, R, F1 = score(out, ref, model_type="bert-base-multilingual-cased", verbose=True)
print("BERTScore")
print(f"System level F1 score: {F1.mean():.4f}")
print(f"System level Precision score: {P.mean():.4f}")
print(f"System level Recall score: {R.mean():.4f}")