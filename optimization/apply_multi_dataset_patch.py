"""
Patch script to add multi-dataset aggregation support to scripts/hyperparameter_optimization.py.

Usage:
    python apply_multi_dataset_patch.py

This script will automatically modify scripts/hyperparameter_optimization.py to support Scheme A (multi-dataset aggregated optimization).
"""

import re
from pathlib import Path


def apply_patch():
    """应用补丁."""
    file_path = Path(__file__).resolve().parents[1] / 'scripts' / 'hyperparameter_optimization.py'
    
    if not file_path.exists():
        print("错误: 找不到 scripts/hyperparameter_optimization.py")
        return False
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份
    backup_path = Path(__file__).resolve().parents[1] / 'scripts' / 'hyperparameter_optimization_before_patch.py'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份原文件到: {backup_path}")
    
    # 应用修改
    modified = content
    changes = []
    
    # 1. 修改docstring
    if '支持两种模式' not in modified:
        old_doc = '''"""超参数优化主脚本.

支持网格搜索和粒子群优化两种超参数搜索方法。
通过调用 run_main.py 的训练逻辑来评估每组超参数的性能。'''
        
        new_doc = '''"""超参数优化主脚本.

支持网格搜索和粒子群优化两种超参数搜索方法。
通过调用 run_main.py 的训练逻辑来评估每组超参数的性能。

支持两种模式:
    - 单数据集优化: 为单个数据集找最优参数
    - 多数据集聚合优化(方案A): 找一组通用参数，在所有数据集上平均性能最优'''
        
        if old_doc in modified:
            modified = modified.replace(old_doc, new_doc)
            changes.append("更新docstring")
    
    # 2. 修改参数解析
    if '--datasets' not in modified:
        # 找到 --dataset 参数定义
        dataset_pattern = r"parser\.add_argument\('--dataset', type=str, required=True,\s+help='数据集名称[^']*'\)"
        if re.search(dataset_pattern, modified):
            new_args = """parser.add_argument('--dataset', type=str, default=None,
                        help='单个数据集名称 (例如: HUST, CALB) - 与--datasets互斥')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='多个数据集名称 (例如: HUST CALB CALCE) - 与--dataset互斥')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'weighted_mean', 'min', 'median'],
                        help='多数据集聚合方式: mean=平均, weighted_mean=加权平均, min=最差性能, median=中位数')
    parser.add_argument('--dataset_weights', type=float, nargs='+', default=None,
                        help='数据集权重 (仅用于weighted_mean聚合)')"""
            modified = re.sub(dataset_pattern, new_args, modified)
            changes.append("添加多数据集参数")
    
    # 3. 添加聚合函数 (在 parse_args() 之后)
    if 'def aggregate_scores' not in modified:
        aggregate_fn = '''

def aggregate_scores(scores: Dict[str, float], 
                     method: str,
                     weights: List[float] = None) -> float:
    """聚合多个数据集的性能分数.
    
    Args:
        scores: 数据集名称到分数的映射
        method: 聚合方法
        weights: 权重列表（仅用于weighted_mean）
        
    Returns:
        聚合后的分数（越小越好）
    """
    score_values = list(scores.values())
    
    if method == 'mean':
        return float(np.mean(score_values))
    elif method == 'weighted_mean':
        if weights is None:
            weights = [1.0] * len(score_values)
        return float(np.average(score_values, weights=weights))
    elif method == 'min':  # 最差性能
        return float(np.max(score_values))
    elif method == 'median':
        return float(np.median(score_values))
    else:
        return float(np.mean(score_values))

'''
        # 在 parse_args() 函数之后插入
        parse_args_end = modified.find('    return parser.parse_args()')
        if parse_args_end != -1:
            next_newline = modified.find('\n', parse_args_end)
            if next_newline != -1:
                modified = modified[:next_newline+1] + aggregate_fn + modified[next_newline+1:]
                changes.append("添加aggregate_scores函数")
    
    # 4. 修改 run_single_experiment 签名
    if 'dataset: str = None' not in modified:
        old_sig = 'def run_single_experiment(params: Dict[str, Any], \n                         args: argparse.Namespace, \n                         trial_id: int) -> Tuple[float, Dict[str, float]]:'
        new_sig = 'def run_single_experiment(params: Dict[str, Any], \n                         args: argparse.Namespace, \n                         trial_id: int,\n                         dataset: str = None) -> Tuple[float, Dict[str, float]]:'
        
        if old_sig in modified:
            modified = modified.replace(old_sig, new_sig)
            changes.append("修改run_single_experiment签名")
    
    # 5. 添加 run_multi_dataset_experiment 函数
    if 'def run_multi_dataset_experiment' not in modified:
        multi_dataset_fn = '''

def run_multi_dataset_experiment(params: Dict[str, Any],
                                 args: argparse.Namespace,
                                 trial_id: int) -> Tuple[float, Dict[str, Any]]:
    """在多个数据集上运行实验并聚合结果.
    
    Args:
        params: 超参数字典
        args: 全局配置参数
        trial_id: 试验编号
        
    Returns:
        聚合后的分数和详细结果字典
    """
    dataset_scores = {}
    dataset_metrics = {}
    
    print(f"\\n{'='*80}")
    print(f"Trial {trial_id} [多数据集]: 在 {len(args.datasets)} 个数据集上评估")
    print(f"数据集: {args.datasets}")
    print(f"参数: {json.dumps(params, indent=2)}")
    print(f"{'='*80}\\n")
    
    # 在每个数据集上评估
    for dataset in args.datasets:
        try:
            score, metrics = run_single_experiment(params, args, trial_id, dataset=dataset)
            dataset_scores[dataset] = score
            dataset_metrics[dataset] = metrics
        except Exception as e:
            print(f"警告: 数据集 {dataset} 训练失败，跳过 - {str(e)}")
            dataset_scores[dataset] = float('inf')
            dataset_metrics[dataset] = {}
    
    # 过滤掉失败的数据集
    valid_scores = {k: v for k, v in dataset_scores.items() if v != float('inf')}
    
    if not valid_scores:
        print(f"错误: Trial {trial_id} 所有数据集都失败")
        return float('inf'), {}
    
    # 聚合分数
    aggregated_score = aggregate_scores(valid_scores, args.aggregation, args.dataset_weights)
    
    print(f"\\nTrial {trial_id} 聚合结果:")
    for dataset, score in valid_scores.items():
        print(f"  {dataset}: {score:.4f}")
    print(f"  聚合分数 ({args.aggregation}): {aggregated_score:.4f}")
    
    # 构建返回结果
    result = {
        'aggregated_score': aggregated_score,
        'dataset_scores': dataset_scores,
        'dataset_metrics': dataset_metrics
    }
    
    return aggregated_score, result

'''
        # 在 parse_training_output 之前插入
        parse_output_start = modified.find('def parse_training_output')
        if parse_output_start != -1:
            modified = modified[:parse_output_start] + multi_dataset_fn + modified[parse_output_start:]
            changes.append("添加run_multi_dataset_experiment函数")
    
    # 6. 修改 main() 函数中的参数验证
    if '必须指定 --dataset 或 --datasets 之一' not in modified:
        main_start = modified.find('def main():')
        if main_start != -1:
            # 找到 args = parse_args() 之后
            args_line = modified.find('args = parse_args()', main_start)
            if args_line != -1:
                next_section = modified.find('# 获取搜索空间', args_line)
                if next_section != -1:
                    validation_code = '''
    
    # 验证参数
    if args.dataset is None and args.datasets is None:
        raise ValueError("必须指定 --dataset 或 --datasets 之一")
    if args.dataset is not None and args.datasets is not None:
        raise ValueError("--dataset 和 --datasets 不能同时指定")
    
    # 标准化为列表
    if args.dataset:
        args.datasets = [args.dataset]
    
    # 验证权重
    if args.aggregation == 'weighted_mean' and args.dataset_weights:
        if len(args.dataset_weights) != len(args.datasets):
            raise ValueError(f"权重数量({len(args.dataset_weights)})必须与数据集数量({len(args.datasets)})一致")
    
    '''
                    modified = modified[:next_section] + validation_code + modified[next_section:]
                    changes.append("添加参数验证逻辑")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified)
    
    if changes:
        print("\n✓ 成功应用以下修改:")
        for change in changes:
            print(f"  - {change}")
        print(f"\n✓ 文件已更新: {file_path}")
        print("\n⚠ 注意: 还需要手动修改以下内容:")
        print("  1. run_single_experiment 函数体中，使用 dataset 参数")
        print("  2. evaluate_fn 中添加多数据集逻辑")
        print("  3. save_results 中处理多数据集结果")
        print("\n详细修改请参考: docs/MULTI_DATASET_OPTIMIZATION.md")
        return True
    else:
        print("✓ 文件已包含所有修改，无需重复应用")
        return True


if __name__ == '__main__':
    print("="*80)
    print("为 scripts/hyperparameter_optimization.py 应用多数据集支持补丁")
    print("="*80 + "\n")
    
    success = apply_patch()
    
    if success:
        print("\n✓ 补丁应用完成！")
        print("\n下一步:")
        print("1. 查看 docs/MULTI_DATASET_OPTIMIZATION.md 了解详细用法")
        print("2. 测试单数据集模式是否仍正常工作:")
        print("   python run.py hyperopt --method pso --model MLP --dataset HUST")
        print("3. 测试多数据集模式:")
        print("   python run.py hyperopt --method pso --model MLP --datasets HUST CALB --aggregation mean")
    else:
        print("\n✗ 补丁应用失败")
