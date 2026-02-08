#!/usr/bin/env python3
"""Calculate average metrics from RAGAS evaluation artifacts."""
import json
import csv
from pathlib import Path

def main():
    artifacts_dir = Path("artifacts")
    
    # Load RAGAS CSV data
    metrics_sums = {}
    count = 0
    
    csv_file = artifacts_dir / "ragas_50_random_ollama.csv"
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            for key in ['recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr', 'bleu_avg', 'rouge1', 'rougeL', 
                        'bertscore_f1', 'grounding_ratio', 'claim_support_ratio', 'faithfulness', 
                        'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness',
                        'answer_similarity', 'confidence', 'correct']:
                try:
                    val = float(row.get(key, '0') or '0')
                    metrics_sums[key] = metrics_sums.get(key, 0) + val
                except:
                    pass
    
    print('=' * 60)
    print('RAGAS 50 Queries Evaluation - Average Metrics')
    print('=' * 60)
    print(f'Total Queries Evaluated: {count}')
    print()
    print('RETRIEVAL METRICS:')
    print(f'  Recall@1:   {metrics_sums.get("recall_at_1", 0)/count:.4f}')
    print(f'  Recall@5:   {metrics_sums.get("recall_at_5", 0)/count:.4f}')
    print(f'  Recall@10:  {metrics_sums.get("recall_at_10", 0)/count:.4f}')
    print(f'  MRR:        {metrics_sums.get("mrr", 0)/count:.4f}')
    print()
    print('RAGAS METRICS:')
    print(f'  Faithfulness:       {metrics_sums.get("faithfulness", 0)/count:.4f}')
    print(f'  Answer Relevancy:   {metrics_sums.get("answer_relevancy", 0)/count:.4f}')
    print(f'  Context Precision:  {metrics_sums.get("context_precision", 0)/count:.4f}')
    print(f'  Context Recall:     {metrics_sums.get("context_recall", 0)/count:.4f}')
    print(f'  Answer Correctness: {metrics_sums.get("answer_correctness", 0)/count:.4f}')
    print(f'  Answer Similarity:  {metrics_sums.get("answer_similarity", 0)/count:.4f}')
    print()
    print('GENERATION QUALITY METRICS:')
    print(f'  BLEU Avg:       {metrics_sums.get("bleu_avg", 0)/count:.4f}')
    print(f'  ROUGE-1:        {metrics_sums.get("rouge1", 0)/count:.4f}')
    print(f'  ROUGE-L:        {metrics_sums.get("rougeL", 0)/count:.4f}')
    print(f'  BERTScore F1:   {metrics_sums.get("bertscore_f1", 0)/count:.4f}')
    print()
    print('GROUNDING/VERIFICATION METRICS:')
    print(f'  Grounding Ratio:      {metrics_sums.get("grounding_ratio", 0)/count:.4f}')
    print(f'  Claim Support Ratio:  {metrics_sums.get("claim_support_ratio", 0)/count:.4f}')
    print()
    print('ACCURACY & CONFIDENCE:')
    print(f'  Accuracy (Correct):  {metrics_sums.get("correct", 0)/count:.4f} ({metrics_sums.get("correct", 0):.0f}/{count})')
    print(f'  Avg Confidence:      {metrics_sums.get("confidence", 0)/count:.4f}')
    print()
    print('=' * 60)
    
    # Return the averages dict for report generation
    return {k: v/count for k, v in metrics_sums.items()}, count

if __name__ == "__main__":
    main()
