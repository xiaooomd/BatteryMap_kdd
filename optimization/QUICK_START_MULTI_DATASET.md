# 多数据集超参数优化快速使用指南

## 🎯 两种方案快速选择

### 方案A: 多数据集聚合优化（通用参数）
找**一组**超参数，在所有数据集上平均性能最优

```bash
# 需要先应用补丁
python apply_multi_dataset_patch.py

# 然后运行
python hyperparameter_optimization.py \
    --method pso \
    --model MLP \
    --datasets HUST CALB CALCE \
    --aggregation mean \
    --gpu 0
```

**输出**: 1组参数，适用于所有数据集

---

### 方案C: 并行独立优化（定制参数）⭐ 推荐
为每个数据集**分别**找最优参数

```bash
# 直接运行
python run_multi_dataset_optimization.py \
    --datasets HUST CALB CALCE MIT MATR SNL ISU_ILCC NA RWTH Stanford XJTU HNEI MICH MICH_EXP UL_PUR Tongji ZNion \
    --method pso \
    --model MLP \
    --n_particles 20 \
    --n_iterations 50 \
    --gpus 0 1 2 3 4 5 \
    --max_workers_per_gpu 4
```

**输出**: 17组参数，每个数据集独立配置

---

## ⚡ 快速测试（3个数据集）

```bash
# 方案C - 快速测试
python run_multi_dataset_optimization.py \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 10 \
    --n_iterations 20 \
    --train_epochs 5 \
    --gpus 0 1 \
    --max_workers_per_gpu 2
```

预计耗时：约1-2小时（取决于GPU性能）

---

## 📊 查看结果

### 方案A结果
```
hyperparam_search_results/
└── pso_MLP_HUST_CALB_CALCE_20260119_220045/
    ├── all_trials.csv           # 所有试验记录
    ├── best_params.json         # 最佳通用参数
    └── search_config.json       # 搜索配置
```

### 方案C结果
```
hyperparam_search_results/
└── multi_dataset_optimization_20260119_220045/
    ├── summary.json             # 汇总结果
    ├── summary.csv              # 汇总表格
    ├── config.json              # 运行配置
    ├── HUST/
    │   ├── all_trials.csv
    │   └── best_params.json
    ├── CALB/
    │   ├── all_trials.csv
    │   └── best_params.json
    └── ... (其他数据集)
```

---

## 🔧 常见参数调整

### 快速测试（降低计算量）
```bash
--n_particles 10        # 从20降到10
--n_iterations 20       # 从50降到20
--train_epochs 5        # 从10降到5
```

### 深度优化（提高精度）
```bash
--n_particles 30        # 从20增到30
--n_iterations 100      # 从50增到100
--train_epochs 20       # 从10增到20
```

### GPU配置
```bash
# 6个GPU，每个最多4任务 = 24并行
--gpus 0 1 2 3 4 5
--max_workers_per_gpu 4

# GPU内存不足时，减少并行任务
--max_workers_per_gpu 2
```

---

## 💡 推荐工作流

### 第1步：快速探索（方案C，少量迭代）
```bash
python run_multi_dataset_optimization.py \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 10 \
    --n_iterations 20 \
    --train_epochs 5 \
    --gpus 0 1 2
```

### 第2步：深度优化（方案C，充分迭代）
```bash
python run_multi_dataset_optimization.py \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 30 \
    --n_iterations 100 \
    --train_epochs 20 \
    --gpus 0 1 2
```

### 第3步（可选）：测试通用性（方案A）
```bash
# 先应用补丁
python apply_multi_dataset_patch.py

# 运行方案A
python hyperparameter_optimization.py \
    --method pso \
    --model MLP \
    --datasets HUST CALB CALCE \
    --aggregation mean \
    --n_particles 20 \
    --n_iterations 50 \
    --train_epochs 10 \
    --gpu 0
```

---

## 🚨 故障排查

### 问题1: 某个数据集总是失败
- 方案C会自动跳过失败的数据集，不影响其他
- 查看该数据集的独立日志
- 可能原因：数据缺失、样本太少、GPU内存不足

### 问题2: GPU内存不足
```bash
--max_workers_per_gpu 2  # 减少并行任务
--batch_size 16          # 减小batch size
```

### 问题3: 进程卡死
- 方案C中，单个数据集卡死不影响其他
- 可以手动kill该进程，其他数据集继续运行

---

## 📚 详细文档

- **完整使用文档**: [docs/MULTI_DATASET_OPTIMIZATION.md](docs/MULTI_DATASET_OPTIMIZATION.md)
- **单数据集优化**: [docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)
- **代码细节**: 查看各脚本的docstring

---

## ⏱️ 耗时估算

### 方案C（17个数据集，6个GPU）
```
单数据集耗时 = 20粒子 × 50迭代 × 5分钟/次 = 83小时
并行加速 = 6GPU × 4任务/GPU = 24并行
实际耗时 ≈ 83小时（因为数据集不会完全同时结束）

建议：先用少量迭代快速测试（耗时约10小时）
```

### 方案A（3个数据集）
```
单次试验 = 3个数据集 × 5分钟 = 15分钟
总耗时 = 20粒子 × 50迭代 × 15分钟 = 250小时

建议：使用网格搜索 + 小搜索空间
```

---

## 🎓 核心概念

- **方案A**: 一组参数 → 所有数据集通用 → 简化部署
- **方案C**: N组参数 → 每个数据集定制 → 最佳性能
- **GPU配置**: 你有6个GPU，每个可跑4个任务 → 最多24并行
- **聚合方式**: mean(平均), min(最差), weighted_mean(加权)
- **失败容错**: 某个数据集失败不影响其他数据集

---

**开始优化吧！** 🚀

有问题请参考: [docs/MULTI_DATASET_OPTIMIZATION.md](docs/MULTI_DATASET_OPTIMIZATION.md)
