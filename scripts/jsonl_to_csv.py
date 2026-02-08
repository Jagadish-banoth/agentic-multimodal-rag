#!/usr/bin/env python3
"""Convert full_eval.jsonl to CSV for demo presentation."""

import json
import csv
import os

def jsonl_to_csv(input_path: str, output_path: str):
    """Convert JSONL evaluation results to a clean CSV for demo."""
    
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from {input_path}")
    
    # Prepare CSV rows with key demo-friendly columns
    csv_rows = []
    for rec in records:
        metrics = rec.get('metrics', {})
        gen_struct = rec.get('generated_structured', {})
        retrieval = metrics.get('retrieval', {})
        generation = metrics.get('generation', {})
        grounding = metrics.get('grounding', {})
        
        # Clean up the generated answer (truncate for readability)
        generated = rec.get('generated', '')
        if len(generated) > 500:
            generated = generated[:500] + '...'
        
        row = {
            'Query ID': rec.get('query_id', ''),
            'Query': rec.get('query', ''),
            'Generated Answer': generated,
            'Reference Answer': metrics.get('reference', ''),
            'Correct': metrics.get('correct', ''),
            'Confidence': gen_struct.get('confidence', ''),
            'Model': gen_struct.get('model', ''),
            'Tokens Used': gen_struct.get('tokens_used', ''),
            # Retrieval metrics
            'Recall@1': retrieval.get('recall@1', ''),
            'Recall@5': retrieval.get('recall@5', ''),
            'Recall@10': retrieval.get('recall@10', ''),
            'MRR': retrieval.get('mrr', ''),
            # Generation metrics
            'BLEU-1': round(generation.get('bleu1', 0), 4) if generation.get('bleu1') else '',
            'BLEU-4': round(generation.get('bleu4', 0), 4) if generation.get('bleu4') else '',
            'ROUGE-1': round(generation.get('rouge1', 0), 4) if generation.get('rouge1') else '',
            'ROUGE-L': round(generation.get('rougeL', 0), 4) if generation.get('rougeL') else '',
            'BERTScore F1': round(generation.get('bertscore_f1', 0), 4) if generation.get('bertscore_f1') else '',
            # Grounding metrics
            'Support Ratio': round(grounding.get('support_ratio', 0), 4) if grounding.get('support_ratio') else '',
            'Claim Support Ratio': round(grounding.get('claim_grounding', {}).get('claim_support_ratio', 0), 4) if grounding.get('claim_grounding', {}).get('claim_support_ratio') else '',
            # Source info
            'Num Retrieved': len(rec.get('retrieved', [])),
        }
        csv_rows.append(row)
    
    # Write CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"CSV written to {output_path}")
        print(f"Columns: {fieldnames}")
    else:
        print("No records to write")


if __name__ == '__main__':
    input_file = 'artifacts/full_eval.jsonl'
    output_file = 'artifacts/full_eval_demo.csv'
    
    if os.path.exists(input_file):
        jsonl_to_csv(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
