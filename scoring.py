import glob
import nltk
import numpy as np
from datasets import load_metric
from bert_score import score

nltk.download('punkt')
metric = load_metric("rouge")

def load_data(path=""):
    # path = "/content/drive/MyDrive/Colab Notebooks/Tesis"

    can_path = path + "ref.txt"
    ref_path = path + "out.txt"
    src_path = path + "src.txt"

    ref = []
    can = []
    src = []

    with open(can_path, 'r') as f:
      for line in f:
        line = line.replace('\n', '')
        can.append(line.strip())

    with open(ref_path, 'r') as f:
      for line in f:
        line = line.replace('\n', '')
        ref.append(line.strip())

    with open(src_path, 'r') as f:
      for line in f:
        line = line.replace('\n', '')
        src.append(line.strip())
    return ref, can, src

def compute_rouge(decoded_preds, decoded_labels):
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    return {k: round(v, 4) for k, v in result.items()}
  
def compute_metrics():
    ref, can, src = load_data()
    
    print("Compute BERTScore...")
    P, R, F1 = score(can, ref, model_type="bert-base-multilingual-cased", verbose=True)
    print(f"System level F1 score: {F1.mean():.4f}")
    print(f"System level Precision score: {P.mean():.4f}")
    print(f"System level Recall score: {R.mean():.4f}")

    print("Compute ROUGE...")
    result = compute_rouge(can, ref)

    print(result)

compute_metrics()