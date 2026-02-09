import subprocess
import sys
import os

# Configuration
models = ['MLP', 'DLinear', 'Autoformer']  # Exclude models starting with CP
datasets = ['HUST']
# Set environment variable to disable wandb
test_env = os.environ.copy()
test_env['WANDB_MODE'] = 'disabled'

args = [
    '--task_type', 'early_prediction',
    '--feature_type', 'extracted_features',
    '--root_path', 'dataset/selected_result/HUST', # 指向具体数据集文件夹
    '--n_selected', '20',
    '--train_epochs', '1',
    '--batch_size', '4',
    '--itr', '1',
    '--checkpoints', './checkpoints_test_smoke',
    '--learning_rate', '0.001'
]

def run_test():
    print("="*60)
    print("🚀 Quick Smoke Test - W&B Disabled")
    print("="*60)

    for model in models:
        for dataset in datasets:
            print(f"\nTesting Model: {model} | Dataset: {dataset}")

            cmd = [sys.executable, 'run_main.py', '--model', model, '--dataset', dataset] + args

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    env=test_env
                )
                print(f"✅ {model} on {dataset}: Success")
                print("Tail output:")
                print('\n'.join(result.stdout.splitlines()[-5:]))

            except subprocess.CalledProcessError as e:
                print(f"❌ {model} on {dataset}: Failed")
                print("Error Output:")
                print(e.stderr)
                # print(e.stdout) # Can also print stdout to help debug
                return False
            except Exception as e:
                print(f"❌ Execution Error: {e}")
                return False

    print("\n="*60)
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    if run_test():
        # 清理
        import shutil
        if os.path.exists('./checkpoints_test_smoke'):
            shutil.rmtree('./checkpoints_test_smoke')
            print("Cleaned up ./checkpoints_test_smoke")
        sys.exit(0)
    else:
        sys.exit(1)
