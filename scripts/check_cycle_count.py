"""Check cycle count statistics for batteries in the li_selected dataset.

Used for diagnosis: whether any battery has insufficient cycles (<100), and to assess impact on training.
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
    """Check cycle counts for all CSV files."""

    if not os.path.exists(root_path):
        safe_print(f"[ERROR] Path does not exist: {root_path}")
        return

    safe_print(f"Scanning: {root_path}\n")

    cycle_stats = defaultdict(list)
    total_files = 0
    problematic_files = []

    # Iterate through all CSV files
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(root, file)
            total_files += 1

            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                num_cycles = len(df)

                cycle_stats['all'].append(num_cycles)

                # Record files with insufficient cycles (<100)
                if num_cycles < 100:
                    problematic_files.append({
                        'file': file,
                        'path': file_path,
                        'cycles': num_cycles
                    })

                # Categorized statistics
                if num_cycles < 50:
                    cycle_stats['<50'].append(num_cycles)
                elif num_cycles < 100:
                    cycle_stats['50-99'].append(num_cycles)
                else:
                    cycle_stats['>=100'].append(num_cycles)

            except Exception as e:
                safe_print(f"[WARN] Cannot read file {file}: {e}")

    # Print statistics
    safe_print(f"{'='*80}")
    safe_print(f"Cycle Count Statistics Report")
    safe_print(f"{'='*80}\n")

    safe_print(f"Total files: {total_files}")
    safe_print(f"Files with insufficient cycles (<100): {len(problematic_files)}\n")

    if cycle_stats['all']:
        all_cycles = cycle_stats['all']
        safe_print(f"Cycle count statistics:")
        safe_print(f"  Min: {min(all_cycles)}")
        safe_print(f"  Max: {max(all_cycles)}")
        safe_print(f"  Mean: {sum(all_cycles) / len(all_cycles):.1f}")
        safe_print(f"  Median: {sorted(all_cycles)[len(all_cycles)//2]}\n")

    safe_print(f"Distribution:")
    safe_print(f"  Cycles < 50:    {len(cycle_stats['<50'])} files")
    safe_print(f"  Cycles 50-99:   {len(cycle_stats['50-99'])} files")
    safe_print(f"  Cycles >= 100:  {len(cycle_stats['>=100'])} files\n")

    # List files with insufficient cycles in detail
    if problematic_files:
        safe_print(f"{'='*80}")
        safe_print(f"Files with insufficient cycles (<100) (Total: {len(problematic_files)})")
        safe_print(f"{'='*80}\n")

        # Sort by cycle count
        problematic_files.sort(key=lambda x: x['cycles'])

        for item in problematic_files[:20]:  # Show only first 20
            safe_print(f"  {item['file']:40s} - {item['cycles']:3d} cycles")

        if len(problematic_files) > 20:
            safe_print(f"\n  ... and {len(problematic_files) - 20} more files not shown")

    # Analyze impact on training
    safe_print(f"\n{'='*80}")
    safe_print(f"Impact Analysis on Training")
    safe_print(f"{'='*80}\n")

    safe_print("[INFO] Key parameters:")
    safe_print(f"  - early_cycle_threshold (default): 100")
    safe_print(f"  - seq_len (default): 5\n")

    safe_print("[INFO] Data loader protection mechanisms:")
    safe_print("  1. In data_loader.py line 985:")
    safe_print("     limit = min(L, self.early_cycle_threshold)")
    safe_print("     -> If cycle count L < 100, only first L cycles are used\n")

    safe_print("  2. In data_loader.py lines 991-994:")
    safe_print("     if limit < start_idx:")
    safe_print("         continue")
    safe_print("     -> If cycles < seq_len (5), skip that battery, no samples generated\n")

    safe_print("  3. Cumulative sample generation (lines 995-1006):")
    safe_print("     for i in range(start_idx, limit + 1):")
    safe_print("     -> Battery with 25 cycles generates 21 samples (i=5 to 25)")
    safe_print("     -> Battery with 100 cycles generates 96 samples (i=5 to 100)\n")

    if problematic_files:
        min_cycles = min(p['cycles'] for p in problematic_files)
        if min_cycles >= 5:
            safe_print(f"[OK] Conclusion: All batteries have cycles >= {min_cycles}, training is normal")
            safe_print(f"   - Batteries with insufficient cycles (<100) will generate fewer training samples")
            safe_print(f"   - But this will not cause errors or training failures")
            safe_print(f"   - Sample count is automatically adjusted based on actual cycle count")
        else:
            safe_print(f"[WARN] Warning: Some batteries have cycles < 5 (min: {min_cycles})")
            safe_print(f"   - These batteries will be skipped, no training samples generated")
            safe_print(f"   - Will not affect training of other batteries")
    else:
        safe_print("[OK] Conclusion: All batteries have cycles >= 100, no issues found")

    safe_print(f"\n{'='*80}\n")


if __name__ == '__main__':
    check_cycle_counts()
