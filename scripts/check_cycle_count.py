"""检查 li_selected 数据集中各电池的循环数统计.

用于诊断是否有循环数不足100的电池，以及评估对训练的影响。
"""

import os
import pandas as pd
from collections import defaultdict


def safe_print(message=""):
    text = str(message)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def check_cycle_counts(root_path='dataset/li_selected_results'):
    """检查所有CSV文件的循环数."""
    
    if not os.path.exists(root_path):
        safe_print(f"[ERROR] 路径不存在: {root_path}")
        return
    
    safe_print(f"正在扫描: {root_path}\n")
    
    cycle_stats = defaultdict(list)
    total_files = 0
    problematic_files = []
    
    # 遍历所有CSV文件
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith('.csv'):
                continue
            
            file_path = os.path.join(root, file)
            total_files += 1
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                num_cycles = len(df)
                
                cycle_stats['all'].append(num_cycles)
                
                # 记录循环数不足100的文件
                if num_cycles < 100:
                    problematic_files.append({
                        'file': file,
                        'path': file_path,
                        'cycles': num_cycles
                    })
                
                # 分类统计
                if num_cycles < 50:
                    cycle_stats['<50'].append(num_cycles)
                elif num_cycles < 100:
                    cycle_stats['50-99'].append(num_cycles)
                else:
                    cycle_stats['>=100'].append(num_cycles)
                    
            except Exception as e:
                safe_print(f"[WARN] 无法读取文件 {file}: {e}")
    
    # 打印统计结果
    safe_print(f"{'='*80}")
    safe_print(f"循环数统计报告")
    safe_print(f"{'='*80}\n")
    
    safe_print(f"总文件数: {total_files}")
    safe_print(f"循环数不足100的文件数: {len(problematic_files)}\n")
    
    if cycle_stats['all']:
        all_cycles = cycle_stats['all']
        safe_print(f"循环数统计:")
        safe_print(f"  最小值: {min(all_cycles)}")
        safe_print(f"  最大值: {max(all_cycles)}")
        safe_print(f"  平均值: {sum(all_cycles) / len(all_cycles):.1f}")
        safe_print(f"  中位数: {sorted(all_cycles)[len(all_cycles)//2]}\n")
    
    safe_print(f"分布统计:")
    safe_print(f"  循环数 < 50:    {len(cycle_stats['<50'])} 个文件")
    safe_print(f"  循环数 50-99:   {len(cycle_stats['50-99'])} 个文件")
    safe_print(f"  循环数 >= 100:  {len(cycle_stats['>=100'])} 个文件\n")
    
    # 详细列出循环数不足100的文件
    if problematic_files:
        safe_print(f"{'='*80}")
        safe_print(f"循环数不足100的文件列表 (共 {len(problematic_files)} 个)")
        safe_print(f"{'='*80}\n")
        
        # 按循环数排序
        problematic_files.sort(key=lambda x: x['cycles'])
        
        for item in problematic_files[:20]:  # 只显示前20个
            safe_print(f"  {item['file']:40s} - {item['cycles']:3d} 个循环")
        
        if len(problematic_files) > 20:
            safe_print(f"\n  ... 还有 {len(problematic_files) - 20} 个文件未显示")
    
    # 分析对训练的影响
    safe_print(f"\n{'='*80}")
    safe_print(f"对训练的影响分析")
    safe_print(f"{'='*80}\n")
    
    safe_print("[INFO] 关键参数:")
    safe_print(f"  - early_cycle_threshold (默认): 100")
    safe_print(f"  - seq_len (默认): 5\n")
    
    safe_print("[INFO] 数据加载器的保护机制:")
    safe_print("  1. 在 data_loader.py 第985行:")
    safe_print("     limit = min(L, self.early_cycle_threshold)")
    safe_print("     -> 如果循环数 L < 100，则只使用前 L 个循环\n")
    
    safe_print("  2. 在 data_loader.py 第991-994行:")
    safe_print("     if limit < start_idx:")
    safe_print("         continue")
    safe_print("     -> 如果循环数 < seq_len (5)，则跳过该电池，不生成样本\n")
    
    safe_print("  3. 累计式样本生成 (第995-1006行):")
    safe_print("     for i in range(start_idx, limit + 1):")
    safe_print("     -> 循环数为25的电池会生成21个样本 (i=5到25)")
    safe_print("     -> 循环数为100的电池会生成96个样本 (i=5到100)\n")
    
    if problematic_files:
        min_cycles = min(p['cycles'] for p in problematic_files)
        if min_cycles >= 5:
            safe_print(f"[OK] 结论: 所有电池的循环数都 >= {min_cycles}，均可正常训练")
            safe_print(f"   - 循环数不足100的电池会生成较少的训练样本")
            safe_print(f"   - 但不会导致报错或训练失败")
            safe_print(f"   - 样本数量会根据实际循环数自动调整")
        else:
            safe_print(f"[WARN] 警告: 存在循环数 < 5 的电池 (最小{min_cycles}个)")
            safe_print(f"   - 这些电池会被跳过，不生成任何训练样本")
            safe_print(f"   - 不会影响其他电池的训练")
    else:
        safe_print("[OK] 结论: 所有电池的循环数都 >= 100，无任何问题")
    
    safe_print(f"\n{'='*80}\n")


if __name__ == '__main__':
    check_cycle_counts()
