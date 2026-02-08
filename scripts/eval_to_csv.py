#!/usr/bin/env python3
"""
Convert RAGAS evaluation JSONL to a clean CSV with essential metrics.
Usage: python scripts/eval_to_csv.py --input artifacts/full_eval_results.jsonl --output artifacts/eval_summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def jsonl_to_csv(input_path: str, output_path: str):
    """Convert JSONL evaluation results to CSV with key metrics."""
    
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        print("No records found in input file!")
        return
    
    # Define CSV columns - essential info for performance analysis
    fieldnames = [
        'query_id',
        'query',
        'generated_answer',
        'reference_answer',
        'correct',
        # Retrieval metrics
        'recall_at_1',
        'recall_at_5', 
        'recall_at_10',
        'mrr',
        # Generation metrics
        'bleu_avg',
        'rouge1',
        'rougeL',
        'bertscore_f1',
        # Grounding
        'grounding_ratio',
        'claim_support_ratio',
        # RAGAS metrics (if available)
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_recall',
        'answer_correctness',
        'answer_similarity',
        # Metadata
        'num_retrieved',
        'confidence',
    ]
    
    rows = []
    for rec in records:
        metrics = rec.get('metrics', {})
        retrieval = metrics.get('retrieval', {})
        generation = metrics.get('generation', {})
        grounding = metrics.get('grounding', {})
        claim_grounding = metrics.get('claim_grounding', {})
        ragas = metrics.get('ragas_metrics', {})
        gen_struct = rec.get('generated_structured', {})
        
        row = {
            'query_id': rec.get('query_id', ''),
            'query': rec.get('query', '')[:200],  # Truncate long queries
            'generated_answer': rec.get('generated', '')[:500],  # Truncate long answers
            'reference_answer': metrics.get('reference', '')[:200],
            'correct': 1 if metrics.get('correct') else 0,
            # Retrieval
            'recall_at_1': retrieval.get('recall@1', 0),
            'recall_at_5': retrieval.get('recall@5', 0),
            'recall_at_10': retrieval.get('recall@10', 0),
            'mrr': retrieval.get('mrr', 0),
            # Generation
            'bleu_avg': generation.get('bleu_avg', 0),
            'rouge1': generation.get('rouge1', 0),
            'rougeL': generation.get('rougeL', 0),
            'bertscore_f1': generation.get('bertscore_f1', 0),
            # Grounding
            'grounding_ratio': grounding.get('support_ratio', 0),
            'claim_support_ratio': claim_grounding.get('claim_support_ratio', 0),
            # RAGAS
            'faithfulness': ragas.get('faithfulness', ''),
            'answer_relevancy': ragas.get('answer_relevancy', ''),
            'context_precision': ragas.get('context_precision', ''),
            'context_recall': ragas.get('context_recall', ''),
            'answer_correctness': ragas.get('answer_correctness', ''),
            'answer_similarity': ragas.get('answer_similarity', ''),
            # Metadata
            'num_retrieved': len(rec.get('retrieved', [])),
            'confidence': gen_struct.get('confidence', 0) if gen_struct else 0,
        }
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Converted {len(rows)} records to {output_path}")
    
    # Print summary stats
    print("\n📊 Summary Statistics:")
    print(f"   Total queries: {len(rows)}")
    print(f"   Correct answers: {sum(r['correct'] for r in rows)} ({100*sum(r['correct'] for r in rows)/len(rows):.1f}%)")
    print(f"   Avg Recall@5: {sum(r['recall_at_5'] for r in rows)/len(rows):.4f}")
    print(f"   Avg MRR: {sum(r['mrr'] for r in rows)/len(rows):.4f}")
    print(f"   Avg ROUGE-L: {sum(r['rougeL'] for r in rows)/len(rows):.4f}")
    print(f"   Avg BERTScore F1: {sum(r['bertscore_f1'] for r in rows)/len(rows):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RAGAS JSONL to CSV')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    args = parser.parse_args()
    
    jsonl_to_csv(args.input, args.output)
