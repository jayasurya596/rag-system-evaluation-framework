import json
import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import DATA_DIR, EVAL_DATASET_PATH, get_api_key
from retriever import HybridRetriever
from rag_pipeline import RAGPipeline
from evaluator import RAGEvaluator

def setup_indices():
    """Build the retrieval indices for both baseline and improved configurations."""
    print("=== STEP 1: Setting up indices ===")
    
    # 1. Build Baseline Index
    print("\n--- Building Baseline Index (Chunk Size: 500) ---")
    baseline_retriever = HybridRetriever(mode="baseline")
    baseline_retriever.build_index()
    
    # 2. Build Improved Index
    print("\n--- Building Improved Index (Chunk Size: 1000) ---")
    improved_retriever = HybridRetriever(mode="improved")
    improved_retriever.build_index()
    
    print("\n[+] Indices setup complete.")

def run_eval_for_pipeline(pipeline, evaluator, test_questions):
    """Run all test questions through the pipeline and return evaluations."""
    results = []
    
    for idx, qa in enumerate(test_questions):
        print(f"  [{idx+1}/{len(test_questions)}] Querying: \"{qa['question']}\" (Category: {qa['category']})")
        
        # Run pipeline
        t0 = time.time()
        res = pipeline.query(qa["question"])
        
        # Run evaluator
        retrieval_metrics = evaluator.evaluate_retrieval(res["retrieved_chunks"], qa["expected_sources"])
        
        # Prepare context text for the judge
        context_str = ""
        for c_idx, chunk in enumerate(res["retrieved_chunks"]):
            context_str += f"[{c_idx}] Source: {chunk['source']}\n{chunk['text']}\n\n"
            
        faith_res = evaluator.evaluate_faithfulness(context_str, res["answer"], qa["question"])
        rel_res = evaluator.evaluate_relevance(qa["question"], res["answer"])
        
        # Accumulate judge costs
        metrics = res["metrics"].copy()
        metrics["judge_cost"] = faith_res.get("judge_cost", 0.0) + rel_res.get("judge_cost", 0.0)
        metrics["total_cost"] = metrics["cost"] + metrics["judge_cost"]
        
        results.append({
            "id": qa["id"],
            "category": qa["category"],
            "question": qa["question"],
            "answer": res["answer"],
            "citations": res["citations"],
            "precision": retrieval_metrics["precision"],
            "recall": retrieval_metrics["recall"],
            "faithfulness": faith_res["faithfulness_score"],
            "relevance": rel_res["relevance_score"],
            "latency": metrics["total_latency"],
            "cost": metrics["total_cost"],
            "retrieved_count": len(res["retrieved_chunks"])
        })
        
        # Throttle slightly to respect Gemini rate limits
        time.sleep(0.3)
        
    return results

def compile_metrics(results):
    """Calculate average metrics overall and per category."""
    df = pd.DataFrame(results)
    
    summary = {
        "overall": {
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "faithfulness": df["faithfulness"].mean(),
            "relevance": df["relevance"].mean(),
            "latency": df["latency"].mean(),
            "cost": df["cost"].mean(),
            "count": len(df)
        }
    }
    
    # Category segmentation
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        summary[cat] = {
            "precision": cat_df["precision"].mean(),
            "recall": cat_df["recall"].mean(),
            "faithfulness": cat_df["faithfulness"].mean(),
            "relevance": cat_df["relevance"].mean(),
            "latency": cat_df["latency"].mean(),
            "cost": cat_df["cost"].mean(),
            "count": len(cat_df)
        }
        
    return summary

def generate_comparison_plots(baseline_summary, improved_summary):
    """Generate charts to visualize baseline vs improved metrics."""
    categories = ["overall", "direct", "multi-hop", "ambiguous", "unanswerable"]
    
    # 1. Retrieval Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Precision
    base_prec = [baseline_summary[cat]["precision"] for cat in categories]
    imp_prec = [improved_summary[cat]["precision"] for cat in categories]
    ax1.bar(x - width/2, base_prec, width, label='Baseline (500 chars, no re-rank)', color='#e74c3c')
    ax1.bar(x + width/2, imp_prec, width, label='Improved (1000 chars, re-ranked)', color='#2ecc71')
    ax1.set_ylabel('Retrieval Precision')
    ax1.set_title('Retrieval Precision by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Recall
    base_rec = [baseline_summary[cat]["recall"] for cat in categories]
    imp_rec = [improved_summary[cat]["recall"] for cat in categories]
    ax2.bar(x - width/2, base_rec, width, label='Baseline', color='#e74c3c')
    ax2.bar(x + width/2, imp_rec, width, label='Improved', color='#2ecc71')
    ax2.set_ylabel('Retrieval Recall')
    ax2.set_title('Retrieval Recall by Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / "retrieval_comparison.png", dpi=150)
    plt.close()
    
    # 2. Generation Quality (Faithfulness and Relevance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Faithfulness
    base_faith = [baseline_summary[cat]["faithfulness"] for cat in categories]
    imp_faith = [improved_summary[cat]["faithfulness"] for cat in categories]
    ax1.bar(x - width/2, base_faith, width, label='Baseline', color='#e74c3c')
    ax1.bar(x + width/2, imp_faith, width, label='Improved', color='#2ecc71')
    ax1.set_ylabel('Faithfulness Score')
    ax1.set_title('Answer Faithfulness by Category (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Relevance
    base_rel = [baseline_summary[cat]["relevance"] for cat in categories]
    imp_rel = [improved_summary[cat]["relevance"] for cat in categories]
    ax2.bar(x - width/2, base_rel, width, label='Baseline', color='#e74c3c')
    ax2.bar(x + width/2, imp_rel, width, label='Improved', color='#2ecc71')
    ax2.set_ylabel('Relevance Score (1-5)')
    ax2.set_title('Answer Relevance by Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / "generation_comparison.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip rebuilding Qdrant indexes")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of questions to test (default 20, use 105 for full set)")
    args = parser.parse_args()
    
    # Load evaluation dataset
    if not EVAL_DATASET_PATH.exists():
        print(f"Evaluation dataset not found at {EVAL_DATASET_PATH}. Please run generate_eval_dataset.py first.")
        return
        
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    # Select sample
    # To ensure balance, sample equally from each category
    categories = ["direct", "multi-hop", "ambiguous", "unanswerable"]
    per_cat = args.sample_size // len(categories)
    
    test_questions = []
    for cat in categories:
        cat_pairs = [qa for qa in qa_pairs if qa["category"] == cat]
        test_questions.extend(cat_pairs[:per_cat])
        
    print(f"Selected {len(test_questions)} questions for evaluation ({per_cat} per category).")
    
    # Step 1: Set up indices unless skipped
    if not args.skip_indexing:
        setup_indices()
    else:
        print("Skipping indexing as requested.")
        
    evaluator = RAGEvaluator()
    
    # Step 2: Validate LLM Judge
    judge_val = evaluator.validate_llm_judge()
    
    # Step 3: Evaluate Baseline Pipeline
    print("\n=== STEP 3: Evaluating Baseline Pipeline ===")
    baseline_pipeline = RAGPipeline(mode="baseline")
    baseline_results = run_eval_for_pipeline(baseline_pipeline, evaluator, test_questions)
    baseline_summary = compile_metrics(baseline_results)
    
    # Step 4: Evaluate Improved Pipeline
    print("\n=== STEP 4: Evaluating Improved Pipeline ===")
    improved_pipeline = RAGPipeline(mode="improved")
    improved_results = run_eval_for_pipeline(improved_pipeline, evaluator, test_questions)
    improved_summary = compile_metrics(improved_results)
    
    # Step 5: Save results & create plots
    generate_comparison_plots(baseline_summary, improved_summary)
    
    report = {
        "judge_validation": judge_val,
        "baseline_summary": baseline_summary,
        "improved_summary": improved_summary,
        "raw_results": {
            "baseline": baseline_results,
            "improved": improved_results
        }
    }
    
    report_path = DATA_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Written full evaluation report to {report_path}")
    
    # Output beautiful Markdown Summary
    print("\n" + "="*50)
    print("             EVALUATION REPORT SUMMARY")
    print("="*50)
    
    metrics = ["precision", "recall", "faithfulness", "relevance", "latency", "cost"]
    
    print("| Metric | Category | Baseline | Improved | Delta |")
    print("| --- | --- | --- | --- | --- |")
    
    for metric in metrics:
        for cat in ["overall"] + categories:
            val_base = baseline_summary[cat][metric]
            val_imp = improved_summary[cat][metric]
            diff = val_imp - val_base
            
            # Format outputs
            if metric in ["precision", "recall", "faithfulness"]:
                fmt_base = f"{val_base*100:.1f}%"
                fmt_imp = f"{val_imp*100:.1f}%"
                fmt_diff = f"{diff*100:+.1f}%"
            elif metric == "relevance":
                fmt_base = f"{val_base:.2f}/5"
                fmt_imp = f"{val_imp:.2f}/5"
                fmt_diff = f"{diff:+.2f}"
            elif metric == "latency":
                fmt_base = f"{val_base:.3f}s"
                fmt_imp = f"{val_imp:.3f}s"
                fmt_diff = f"{diff:+.3f}s"
            else: # cost
                fmt_base = f"${val_base:.6f}"
                fmt_imp = f"${val_imp:.6f}"
                fmt_diff = f"${diff:+.6f}"
                
            print(f"| {metric.capitalize()} | {cat.upper()} | {fmt_base} | {fmt_imp} | {fmt_diff} |")
            
    print("\nJudge Validation Accuracy: ", f"{judge_val['accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()
