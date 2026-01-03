"""Run all MixMamba experiments for 1-shot, 5-shot, 10-shot with various training sample sizes."""
import subprocess
import sys
import argparse
import os
import re

def get_args():
    parser = argparse.ArgumentParser(description='Run all MixMamba experiments')
    parser.add_argument('--project', type=str, default='mixmamba-fewshot', help='WandB project name')
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/test/scalogram_minh',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='scalogram_minh', help='Dataset name for logging')
    return parser.parse_args()


# Configuration
SHOTS = [1, 5, 10]

# Training samples per class: [min for 10-shot, medium, large, all]
# Note: 80 total samples = 20 per class (since way_num=4)
SAMPLES_LIST = [80, 800, 1600, 6000]

# Model variants (MixMamba uses CovarianceNet)
MODELS = ['mixmamba']


def run_experiment(model, shot, samples, dataset_path, dataset_name, project):
    """Run a single MixMamba experiment."""
    print(f"\n{'='*60}")
    print(f"Model={model}, Shot={shot}, Samples={samples if samples else 'All'}")
    print('='*60)
    
    cmd = [
        sys.executable, 'main.py',
        '--shot_num', str(shot),
        '--way_num', '4',
        '--image_size', '64',
        '--mode', 'train',
        '--project', project,
        '--dataset_path', dataset_path,
        '--dataset_name', dataset_name,
        '--num_epochs', '100',
        '--lr', '1e-3',
        '--episode_num_train', '100',
        '--episode_num_val', '200',
        '--episode_num_test', '300',
    ]
    
    if samples is not None:
        cmd.extend(['--training_samples', str(samples)])
    
    try:
        # Run command and stream output
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def generate_comparison_charts(dataset_name):
    """Generate comparison bar charts from results."""
    try:
        from function.function import plot_model_comparison_bar
    except ImportError:
        print("Warning: Could not import plot function, skipping charts")
        return
    
    results_dir = 'results/'
    
    # Model display names
    model_display_names = {
        'mixmamba': 'MixMamba'
    }
    
    for samples in SAMPLES_LIST:
        samples_str = f"{samples}samples"
        
        model_results = {}
        
        for model in MODELS:
            display_name = model_display_names.get(model, model)
            model_results[display_name] = {}
            
            for shot in SHOTS:
                # Filename format from main.py:
                # results_{dataset_name}_{model}_{samples_str}_{shot}shot.txt
                # Note: main.py uses 'mixmamba' in filename if dataset_name is prefix?
                # Actually main.py saves as: results_{dataset_name}_{samples_str}_{shot}shot.txt
                # Wait, main.py code: 
                # txt_path = os.path.join(args.path_results, f"results_{args.dataset_name}_{samples_str.strip('_')}_{args.shot_num}shot.txt")
                
                # So it does NOT include model name in filename currently for MixMamba main.py
                # This is different from mamba_glscnet which takes --model argument.
                # MixMamba has only one model essentially.
                
                result_file = os.path.join(
                    results_dir,
                    f"results_{dataset_name}_{samples_str}_{shot}shot.txt"
                )
                
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Parse accuracy: Accuracy : 0.9667 ± 0.0123
                        match = re.search(r'Accuracy\s*:\s*([\d.]+)\s*±', content)
                        if match:
                            acc = float(match.group(1))
                            model_results[display_name][f'{shot}shot'] = acc
        
        # Remove incomplete results
        complete_results = {}
        for model, shots_dict in model_results.items():
            # Check if we have both 1shot and 5shot (10shot might not be in plot function)
            # The plot_model_comparison_bar function expects '1shot' and '5shot' keys
            if '1shot' in shots_dict and '5shot' in shots_dict:
                complete_results[model] = shots_dict
        
        if complete_results:
            save_path = os.path.join(results_dir, f"mixmamba_comparison_{dataset_name}_{samples_str}.png")
            try:
                plot_model_comparison_bar(complete_results, samples, save_path)
                print(f"  Chart saved: {save_path}")
            except Exception as e:
                print(f"  Error generating chart: {e}")
        else:
            print(f"  No complete results (1-shot & 5-shot) for {samples_str}")


def main():
    args = get_args()
    
    # Count total experiments
    total_experiments = len(MODELS) * len(SHOTS) * len(SAMPLES_LIST)
    current = 0
    
    print("="*60)
    print("MixMamba - Full Experiment Suite")
    print("="*60)
    print(f"Models: {MODELS}")
    print(f"Shots: {SHOTS}")
    print(f"Training samples: {SAMPLES_LIST}")
    print(f"Dataset: {args.dataset_path} ({args.dataset_name})")
    print(f"Total experiments: {total_experiments}")
    print("="*60)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    success_count = 0
    failed_experiments = []
    
    # Run all experiments
    for model in MODELS:
        for shot in SHOTS:
            for samples in SAMPLES_LIST:
                current += 1
                print(f"\n[{current}/{total_experiments}]", end=" ")
                
                success = run_experiment(
                    model=model,
                    shot=shot,
                    samples=samples,
                    dataset_path=args.dataset_path,
                    dataset_name=args.dataset_name,
                    project=args.project
                )
                
                if success:
                    success_count += 1
                else:
                    failed_experiments.append(f"{model}_{shot}shot_{samples}samples")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total: {total_experiments}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print("\n" + "="*60)
    print("Generating comparison charts...")
    print("="*60)
    
    # Generate comparison after all experiments
    generate_comparison_charts(args.dataset_name)
    
    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()
