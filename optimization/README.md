# Optimization 文件夹

此文件夹包含超参数优化相关的辅助文件和文档。

## 📁 文件说明

### 辅助工具
- **`apply_multi_dataset_patch.py`**: 方案A补丁工具
  - 为 `scripts/hyperparameter_optimization.py` 添加多数据集聚合支持
  - 自动备份原文件
  - 使用方法: `python optimization/apply_multi_dataset_patch.py`

### 文档
- **`QUICK_START_MULTI_DATASET.md`**: 快速使用指南
  - 两种方案的快速选择和示例命令
  - 常见参数调整和故障排查
  
- **`MULTI_DATASET_OPTIMIZATION_SUMMARY.md`**: 完成总结
  - 功能概览和关键要点
  - 推荐工作流
  - 文档导航

## 🚀 主要运行入口

### 单数据集或多数据集聚合（方案A）
```bash
python run.py hyperopt --method pso --model MLP --dataset HUST
```

### 多数据集并行独立优化（方案C）⭐ 推荐
```bash
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE \
    --method pso --model MLP \
    --gpus 0 1 2
```

### 交互式启动
```bash
python run.py hyperopt-batch
```

## 📚 完整文档

详细文档在 `docs/` 文件夹：
- `docs/HYPERPARAMETER_OPTIMIZATION.md` - 单数据集优化详细文档
- `docs/MULTI_DATASET_OPTIMIZATION.md` - 多数据集优化完整指南

## 📦 相关文件位置

- **核心算法**: `utils/optimization.py`
  - GridSearchOptimizer
  - PSOOptimizer
  - RandomSearchOptimizer

- **搜索空间配置**: `configs/hyperparam_search_config.py`
  - 预定义的搜索空间
  - 所有模型的参数范围

- **示例脚本**: `train_eval_scripts/hyperparam_search_examples.sh`
  - Bash脚本示例

## 💡 快速开始

1. **测试单数据集优化**
   ```bash
   python run.py hyperopt \
       --method pso --model MLP --dataset HUST \
       --n_particles 10 --n_iterations 20
   ```

2. **测试多数据集并行优化** ⭐
   ```bash
   python run.py multi-dataset-opt \
       --datasets HUST CALB CALCE \
       --method pso --model MLP \
       --n_particles 10 --n_iterations 20 \
       --gpus 0 1 2
   ```

3. **查看快速指南**
   ```bash
   cat optimization/QUICK_START_MULTI_DATASET.md
   ```

## 🔗 相关链接

- [快速使用指南](QUICK_START_MULTI_DATASET.md)
- [功能总结](MULTI_DATASET_OPTIMIZATION_SUMMARY.md)
- [完整文档](../docs/MULTI_DATASET_OPTIMIZATION.md)
