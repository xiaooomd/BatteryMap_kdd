"""检查 li_selected 数据集中各电池的循环数统计.

用于诊断是否有循环数不足100的电池，以及评估对训练的影响。
"""

import os
import pandas as pd
from collections import defaultdict


def check_cycle_counts(root_path='dataset/li_selected_results'):
    """检查所有CSV文件的循环数."""
    
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return
    
    print(f"正在扫描: {root_path}\n")
    
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
                print(f"⚠️  无法读取文件 {file}: {e}")
    
    # 打印统计结果
    print(f"{'='*80}")
    print(f"循环数统计报告")
    print(f"{'='*80}\n")
    
    print(f"总文件数: {total_files}")
    print(f"循环数不足100的文件数: {len(problematic_files)}\n")
    
    if cycle_stats['all']:
        all_cycles = cycle_stats['all']
        print(f"循环数统计:")
        print(f"  最小值: {min(all_cycles)}")
        print(f"  最大值: {max(all_cycles)}")
        print(f"  平均值: {sum(all_cycles) / len(all_cycles):.1f}")
        print(f"  中位数: {sorted(all_cycles)[len(all_cycles)//2]}\n")
    
    print(f"分布统计:")
    print(f"  循环数 < 50:    {len(cycle_stats['<50'])} 个文件")
    print(f"  循环数 50-99:   {len(cycle_stats['50-99'])} 个文件")
    print(f"  循环数 >= 100:  {len(cycle_stats['>=100'])} 个文件\n")
    
    # 详细列出循环数不足100的文件
    if problematic_files:
        print(f"{'='*80}")
        print(f"循环数不足100的文件列表 (共 {len(problematic_files)} 个)")
        print(f"{'='*80}\n")
        
        # 按循环数排序
        problematic_files.sort(key=lambda x: x['cycles'])
        
        for item in problematic_files[:20]:  # 只显示前20个
            print(f"  {item['file']:40s} - {item['cycles']:3d} 个循环")
        
        if len(problematic_files) > 20:
            print(f"\n  ... 还有 {len(problematic_files) - 20} 个文件未显示")
    
    # 分析对训练的影响
    print(f"\n{'='*80}")
    print(f"对训练的影响分析")
    print(f"{'='*80}\n")
    
    print("📌 关键参数:")
    print(f"  - early_cycle_threshold (默认): 100")
    print(f"  - seq_len (默认): 5\n")
    
    print("✅ 数据加载器的保护机制:")
    print("  1. 在 data_loader.py 第985行:")
    print("     limit = min(L, self.early_cycle_threshold)")
    print("     → 如果循环数 L < 100，则只使用前 L 个循环\n")
    
    print("  2. 在 data_loader.py 第991-994行:")
    print("     if limit < start_idx:")
    print("         continue")
    print("     → 如果循环数 < seq_len (5)，则跳过该电池，不生成样本\n")
    
    print("  3. 累计式样本生成 (第995-1006行):")
    print("     for i in range(start_idx, limit + 1):")
    print("     → 循环数为25的电池会生成21个样本 (i=5到25)")
    print("     → 循环数为100的电池会生成96个样本 (i=5到100)\n")
    
    if problematic_files:
        min_cycles = min(p['cycles'] for p in problematic_files)
        if min_cycles >= 5:
            print(f"✅ 结论: 所有电池的循环数都 >= {min_cycles}，均可正常训练")
            print(f"   - 循环数不足100的电池会生成较少的训练样本")
            print(f"   - 但不会导致报错或训练失败")
            print(f"   - 样本数量会根据实际循环数自动调整")
        else:
            print(f"⚠️  警告: 存在循环数 < 5 的电池 (最小{min_cycles}个)")
            print(f"   - 这些电池会被跳过，不生成任何训练样本")
            print(f"   - 不会影响其他电池的训练")
    else:
        print("✅ 结论: 所有电池的循环数都 >= 100，无任何问题")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    check_cycle_counts()
