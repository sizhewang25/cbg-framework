#!/usr/bin/env python3
"""
Analyze classification table to show success/failure distribution patterns.

Categories:
1. All success targets (shortest_ping=True AND all CBG=True)
2. All failure targets (shortest_ping=False AND all CBG=False)
3. shortest_ping success but all CBG failures (shortest_ping=True AND all CBG=False)
4. shortest_ping failure but at least one CBG succeeds (shortest_ping=False AND >=1 CBG=True)
5. Others (mixed results)
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_classification_table(csv_path: str) -> pd.DataFrame:
    """Load classification table from CSV."""
    return pd.read_csv(csv_path)


def categorize_targets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize targets by success/failure patterns."""
    
    cbg_cols = ["million_scale_cbg", "octant_cbg_hull", "octant_cbg_spline", "spotter_cbg", "vanilla_cbg"]
    categories = {
        "all_success": [],
        "all_failure": [],
        "ping_success_cbg_all_fail": [],
        "ping_fail_cbg_at_least_one_succeed": [],
        "others": []
    }
    
    for idx, row in df.iterrows():
        target_id = row["target_id"]
        ping_success = row["shortest_ping"]
        cbg_results = [row[col] for col in cbg_cols]
        cbg_all_success = all(cbg_results)
        cbg_all_failure = not any(cbg_results)
        cbg_at_least_one_success = any(cbg_results)
        
        # Categorize
        if ping_success and cbg_all_success:
            categories["all_success"].append(target_id)
        elif not ping_success and cbg_all_failure:
            categories["all_failure"].append(target_id)
        elif ping_success and cbg_all_failure:
            categories["ping_success_cbg_all_fail"].append(target_id)
        elif not ping_success and cbg_at_least_one_success:
            categories["ping_fail_cbg_at_least_one_succeed"].append(target_id)
        else:
            categories["others"].append(target_id)
    
    return categories


def print_distribution(categories: Dict[str, List[str]]) -> None:
    """Print distribution statistics."""
    
    print("\n" + "=" * 80)
    print("SUCCESS/FAILURE DISTRIBUTION ANALYSIS")
    print("=" * 80 + "\n")
    
    total = sum(len(targets) for targets in categories.values())
    
    category_names = {
        "all_success": "1. All Success (shortest_ping=True + all CBG=True)",
        "all_failure": "2. All Failure (shortest_ping=False + all CBG=False)",
        "ping_success_cbg_all_fail": "3. Shortest-Ping Success, All CBG Failures",
        "ping_fail_cbg_at_least_one_succeed": "4. Shortest-Ping Failure, At Least One CBG Success",
        "others": "5. Others (mixed results)"
    }
    
    print(f"Total targets: {total}\n")
    
    for key, name in category_names.items():
        targets = categories[key]
        count = len(targets)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"{name}")
        print(f"  Count: {count:3d} ({pct:5.1f}%)")
        if count <= 20:
            print(f"  Targets: {targets}")
        else:
            print(f"  Targets (first 10): {targets[:10]}")
            print(f"              (last 5): {targets[-5:]}")
        print()


def save_category_lists(categories: Dict[str, List[str]], output_dir: str) -> None:
    """Save category lists to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for category_key, targets in categories.items():
        csv_file = output_path / f"{category_key}.csv"
        df = pd.DataFrame({"target_id": targets})
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_success_failure_distribution.py <classification_table_csv> [output_dir]")
        print(f"Example: python {sys.argv[0]} scripts/analysis/outputs/config-test03/cluster/config-test03_classification_table.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(csv_path).parent)
    
    # Load and analyze
    df = load_classification_table(csv_path)
    categories = categorize_targets(df)
    
    # Print results
    print_distribution(categories)
    
    # Save category lists
    print("=" * 80)
    print("Saving category lists to CSV files...")
    print("=" * 80 + "\n")
    save_category_lists(categories, output_dir)


if __name__ == "__main__":
    main()
