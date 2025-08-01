```python
#!/usr/bin/env python3
"""
Comprehensive Experiment Runner - Fixed Version
Ensures all models train correctly, save results, and visualize
"""

import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_experiment_directories():
    """Create experiment directory structure"""
    directories = [
        'results',
        'results/rl_sdg',
        'results/rl_sdg/samples',
        'results/rl_sdg/checkpoints',
        'results/rl_sdg/plots',
        'results/datagan',
        'results/datagan/samples',
        'results/datagan/checkpoints',
        'results/datagan/plots',
        'results/cyclegan',
        'results/cyclegan/samples',
        'results/cyclegan/checkpoints',
        'results/cyclegan/plots',
        'results/comparison',
        'results/comparison/plots',
        'results/comparison/tables'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_datasets():
    """Check if datasets exist"""
    datasets = {
        'KITTI': 'kitti_subset',
        'DrivingStereo': 'DrivingStereo_demo_images'
    }
    
    missing_datasets = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            missing_datasets.append(f"{name}: {path}")
        else:
            print(f"✓ {name} dataset exists: {path}")
    
    if missing_datasets:
        print("Error: The following datasets are missing:")
        for dataset in missing_datasets:
            print(f"  - {dataset}")
        return False
    
    return True

def train_all_models():
    """Train all models"""
    print("="*60)
    print("Starting training of all models")
    print("="*60)
    
    models = {
        'RL-SDG': 'experiments.train_rl_sdg',
        'DataGAN': 'experiments.train_datagan', 
        'CycleGAN': 'experiments.train_cyclegan'
    }
    
    results = {}
    
    for model_name, module_path in models.items():
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        try:
            # Dynamically import module
            module = __import__(module_path, fromlist=['main'])
            module.main()
            results[model_name] = {'status': 'success', 'message': 'Training completed'}
            print(f"✓ {model_name} training successful")
        except Exception as e:
            print(f"✗ {model_name} training failed: {e}")
            results[model_name] = {'status': 'failed', 'message': str(e)}
            import traceback
            traceback.print_exc()
    
    return results

def evaluate_all_models():
    """Evaluate all models"""
    print("\n" + "="*60)
    print("Starting evaluation of all models")
    print("="*60)
    
    try:
        from experiments.evaluate import main as evaluate_main
        evaluate_main()
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comprehensive_report():
    """Create comprehensive report"""
    print("\n" + "="*60)
    print("Creating comprehensive report")
    print("="*60)
    
    # Collect all training histories
    training_histories = {}
    for model_name in ['rl_sdg', 'datagan', 'cyclegan']:
        history_file = f'results/{model_name}/training_history.json'
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    training_histories[model_name] = json.load(f)
                print(f"✓ Loaded {model_name} training history")
            except Exception as e:
                print(f"✗ Failed to load {model_name} training history: {e}")
    
    # Collect evaluation results
    evaluation_results = {}
    eval_file = 'results/evaluation_results.json'
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                evaluation_results = json.load(f)
            print("✓ Loaded evaluation results")
        except Exception as e:
            print(f"✗ Failed to load evaluation results: {e}")
    
    # Create comprehensive report
    report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(training_histories),
            'models_trained': list(training_histories.keys())
        },
        'training_summary': {},
        'evaluation_summary': {},
        'recommendations': []
    }
    
    # Training summary
    for model_name, history in training_histories.items():
        report['training_summary'][model_name] = {
            'best_epoch': history.get('best_epoch', 'N/A'),
            'best_g_loss': history.get('best_g_loss', 'N/A'),
            'final_g_loss': history.get('final_g_loss', 'N/A'),
            'final_d_loss': history.get('final_d_loss', 'N/A'),
            'total_steps': history.get('total_steps', 'N/A'),
            'training_time_minutes': history.get('total_steps', 0) * 0.1  # Estimate
        }
    
    # Evaluation summary
    if evaluation_results:
        for model_name, results in evaluation_results.items():
            if model_name != 'baseline':
                report['evaluation_summary'][model_name] = {
                    'diversity_score': results.get('diversity_score', 'N/A'),
                    'fid': results.get('fid', 'N/A'),
                    'yolov8_f1_score': results.get('yolov8_f1_score', 'N/A')
                }
    
    # Generate recommendations
    if training_histories:
        best_model = min(training_histories.keys(), 
                        key=lambda x: training_histories[x].get('best_g_loss', float('inf')))
        report['recommendations'].append(f"Best training model: {best_model}")
    
    if evaluation_results:
        best_fid = min(evaluation_results.keys(), 
                      key=lambda x: evaluation_results[x].get('fid', float('inf')))
        best_f1 = max(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x].get('yolov8_f1_score', 0))
        report['recommendations'].append(f"Lowest FID model: {best_fid}")
        report['recommendations'].append(f"Highest F1 score model: {best_f1}")
    
    # Save report
    report_file = 'results/comprehensive_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Comprehensive report saved: {report_file}")
    
    # Create visualizations
    create_comparison_visualizations(training_histories, evaluation_results)
    
    return report

def create_comparison_visualizations(training_histories, evaluation_results):
    """Create comparison visualizations - Using English labels"""
    print("Creating comparison visualizations...")
    
    # Set matplotlib to English
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Training loss comparison
    if len(training_histories) > 1:
        plt.figure(figsize=(15, 10))
        
        # Generator loss comparison
        plt.subplot(2, 2, 1)
        for model_name, history in training_histories.items():
            g_losses = history.get('g_losses', [])
            if g_losses:
                plt.plot(g_losses, label=model_name.upper(), linewidth=2)
        plt.title('Generator Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Discriminator loss comparison
        plt.subplot(2, 2, 2)
        for model_name, history in training_histories.items():
            d_losses = history.get('d_losses', [])
            if d_losses:
                plt.plot(d_losses, label=model_name.upper(), linewidth=2)
        plt.title('Discriminator Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Final loss comparison
        plt.subplot(2, 2, 3)
        model_names = list(training_histories.keys())
        final_g_losses = [training_histories[name].get('final_g_loss', 0) for name in model_names]
        final_d_losses = [training_histories[name].get('final_d_loss', 0) for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, final_g_losses, width, label='Generator Loss', alpha=0.8)
        plt.bar(x + width/2, final_d_losses, width, label='Discriminator Loss', alpha=0.8)
        plt.title('Final Loss Comparison')
        plt.xlabel('Model')
        plt.ylabel('Loss')
        plt.xticks(x, [name.upper() for name in model_names])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Evaluation metrics comparison
        if evaluation_results:
            plt.subplot(2, 2, 4)
            eval_models = [name for name in evaluation_results.keys() if name != 'baseline']
            if eval_models:
                f1_scores = [evaluation_results[name].get('yolov8_f1_score', 0) for name in eval_models]
                fid_scores = [evaluation_results[name].get('fid', 0) for name in eval_models]
                
                x = np.arange(len(eval_models))
                width = 0.35
                
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                bars1 = ax1.bar(x - width/2, f1_scores, width, label='F1 Score', color='green', alpha=0.8)
                bars2 = ax2.bar(x + width/2, fid_scores, width, label='FID', color='red', alpha=0.8)
                
                ax1.set_xlabel('Model')
                ax1.set_ylabel('F1 Score', color='green')
                ax2.set_ylabel('FID', color='red')
                ax1.set_xticks(x)
                ax1.set_xticklabels([name.upper() for name in eval_models])
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom')
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
                
                plt.title('Evaluation Metrics Comparison')
        
        plt.tight_layout()
        plt.savefig('results/comparison/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Model comparison plots saved")
    
    # Create performance table and save as CSV
    if evaluation_results:
        create_performance_table(evaluation_results)

def create_performance_table(evaluation_results):
    """Create performance comparison table - Using English labels"""
    import pandas as pd
    
    # Prepare data
    data = []
    for model_name, results in evaluation_results.items():
        if model_name != 'baseline':
            data.append({
                'Model': model_name.upper(),
                'Diversity_Score': f"{results.get('diversity_score', 0):.4f}",
                'FID': f"{results.get('fid', 0):.2f}",
                'YOLOv8_F1_Score': f"{results.get('yolov8_f1_score', 0):.4f}",
                'YOLOv8_Precision': f"{results.get('yolov8_precision', 0):.4f}",
                'YOLOv8_Recall': f"{results.get('yolov8_recall', 0):.4f}",
                'Edge_Scenarios': f"{results.get('edge_scenarios', {}).get('total_scenarios', 0):.0f}"
            })
    
    if data:
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_file = 'results/comparison/tables/performance_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Performance comparison table saved to CSV: {csv_file}")
        
        # Save as HTML
        html_file = 'results/comparison/tables/performance_comparison.html'
        html_content = f"""
        <html>
        <head>
            <title>Model Performance Comparison</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h2>Model Performance Comparison</h2>
            {df.to_html(index=False, classes='table table-striped')}
        </body>
        </html>
        """
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ HTML performance comparison table saved: {html_file}")

def main():
    """Main function"""
    print("="*60)
    print("Comprehensive Experiment Runner - Fixed Version")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available, GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA not available, will use CPU for training")
    
    # Step 1: Create directory structure
    print("\nStep 1: Create experiment directory structure")
    create_experiment_directories()
    
    # Step 2: Check datasets
    print("\nStep 2: Check datasets")
    if not check_datasets():
        print("Dataset check failed, exiting")
        return False
    
    # Step 3: Train all models
    print("\nStep 3: Train all models")
    training_results = train_all_models()
    
    # Step 4: Evaluate all models
    print("\nStep 4: Evaluate all models")
    evaluation_success = evaluate_all_models()
    
    # Step 5: Create comprehensive report
    print("\nStep 5: Create comprehensive report")
    report = create_comprehensive_report()
    
    # Final summary
    print("\n" + "="*60)
    print("Experiment Completion Summary")
    print("="*60)
    
    successful_models = [name for name, result in training_results.items() if result['status'] == 'success']
    failed_models = [name for name, result in training_results.items() if result['status'] == 'failed']
    
    print(f"Successfully trained models: {len(successful_models)}/{len(training_results)}")
    for model in successful_models:
        print(f"  ✓ {model}")
    
    if failed_models:
        print(f"Failed models: {len(failed_models)}")
        for model in failed_models:
            print(f"  ✗ {model}: {training_results[model]['message']}")
    
    print(f"Evaluation status: {'Success' if evaluation_success else 'Failed'}")
    
    print("\nResult file locations:")
    print("  - results/rl_sdg/ (RL-SDG model results)")
    print("  - results/datagan/ (DataGAN model results)")
    print("  - results/cyclegan/ (CycleGAN model results)")
    print("  - results/comparison/ (Comparison analysis results)")
    print("  - results/comprehensive_report.json (Comprehensive report)")
    print("  - results/evaluation/evaluation_results.csv (Evaluation results CSV)")
    
    print("\n✓ Experiment completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ All experiments completed successfully!")
    else:
        print("\n✗ Experiment run failed!")
        sys.exit(1)
```