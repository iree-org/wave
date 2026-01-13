import csv
import sys
import os
from collections import defaultdict
import statistics

def calculate_moe_runtimes(csv_file):
    """
    Calculate runtime in milliseconds for kernels with 'moe' in their name
    """
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return None, 0

    moe_kernels = []
    total_kernels = 0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_kernels += 1
            if 'moe' in row['Kernel_Name'].lower():
                start_ts = int(row['Start_Timestamp'])
                end_ts = int(row['End_Timestamp'])
                runtime_ts = end_ts - start_ts
                runtime_ms = runtime_ts / 1_000_000  # Convert to milliseconds

                kernel_data = {
                    'Kernel_Name': row['Kernel_Name'],
                    'Runtime_timestamps': runtime_ts,
                    'Runtime_ms': runtime_ms,
                    'Start_Timestamp': start_ts,
                    'End_Timestamp': end_ts,
                    'Dispatch_Id': row['Dispatch_Id']
                }
                moe_kernels.append(kernel_data)

    return moe_kernels, total_kernels

def group_kernels_by_name(moe_kernels):
    """
    Group kernels by name and calculate statistics
    """
    grouped = defaultdict(list)

    for kernel in moe_kernels:
        kernel_name = kernel['Kernel_Name']
        grouped[kernel_name].append(kernel['Runtime_ms'])

    return grouped

def calculate_statistics(runtimes):
    """
    Calculate statistics for a list of runtimes
    """
    if not runtimes:
        return None

    return {
        'count': len(runtimes),
        'mean': statistics.mean(runtimes),
        'median': statistics.median(runtimes),
        'min': min(runtimes),
        'max': max(runtimes),
        'stdev': statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        'all_runtimes': runtimes
    }

def print_detailed_results(moe_kernels):
    """
    Print all individual kernel executions
    """
    print("DETAILED KERNEL EXECUTIONS:")
    print("=" * 80)
    for i, kernel in enumerate(moe_kernels, 1):
        print(f"{i:2d}. {kernel['Kernel_Name']}")
        print(f"     Dispatch: {kernel['Dispatch_Id']}")
        print(f"     Runtime:  {kernel['Runtime_ms']:.3f} ms ({kernel['Runtime_timestamps']:,} timestamps)")
        print(f"     Start:    {kernel['Start_Timestamp']:,}")
        print(f"     End:      {kernel['End_Timestamp']:,}")
        print()

def print_grouped_statistics(grouped_kernels, num_warmup=0):
    """
    Print statistics grouped by kernel name
    """
    print("\nGROUPED KERNEL STATISTICS:")
    print("=" * 120)

    # Sort by kernel name for consistent output
    sorted_kernels = sorted(grouped_kernels.items())

    for kernel_name, runtimes in sorted_kernels:
        # Skip warmup iterations if specified
        if num_warmup > 0 and len(runtimes) > num_warmup:
            stable_runtimes = runtimes[num_warmup:]
            print(f"\n{kernel_name}")
            print(f"  Total executions: {len(runtimes)} (showing stats for last {len(stable_runtimes)} after {num_warmup} warmup)")
        else:
            stable_runtimes = runtimes
            print(f"\n{kernel_name}")
            print(f"  Total executions: {len(runtimes)}")

        stats = calculate_statistics(stable_runtimes)

        if stats:
            print(f"  Median:  {stats['median']:.3f} ms")
            print(f"  Mean:    {stats['mean']:.3f} ms")
            print(f"  Min:     {stats['min']:.3f} ms")
            print(f"  Max:     {stats['max']:.3f} ms")
            print(f"  Std Dev: {stats['stdev']:.3f} ms")

            # Show all runtimes for inspection
            print(f"  All times: {', '.join([f'{t:.3f}' for t in stable_runtimes])} ms")

def print_comparison(grouped_kernels, num_warmup=0):
    """
    Print a comparison table of all kernels
    """
    print("\n\nKERNEL COMPARISON (after warmup):")
    print("=" * 100)
    print(f"{'Kernel Name':<100} {'Count':>6} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
    print("-" * 100)

    sorted_kernels = sorted(grouped_kernels.items())
    baseline_median = None

    for kernel_name, runtimes in sorted_kernels:
        # Skip warmup iterations
        if num_warmup > 0 and len(runtimes) > num_warmup:
            stable_runtimes = runtimes[num_warmup:]
        else:
            stable_runtimes = runtimes

        stats = calculate_statistics(stable_runtimes)

        if stats:
            # Truncate kernel name if too long
            display_name = kernel_name if len(kernel_name) <= 100 else kernel_name[:97] + "..."

            print(f"{display_name:<100} {stats['count']:>6} {stats['median']:>9.3f}  {stats['mean']:>9.3f}  "
                  f"{stats['min']:>9.3f}  {stats['max']:>9.3f}  {stats['stdev']:>9.3f}")

            # Track baseline (first kernel, usually Triton)
            if baseline_median is None:
                baseline_median = stats['median']
            else:
                slowdown = (stats['median'] / baseline_median - 1) * 100
                print(f"{'':>100} {'':>6} {slowdown:>9.1f}% slower than baseline")

def main():
    csv_file = sys.argv[1]
    num_warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if num_warmup > 0:
        print(f"Skipping first {num_warmup} iterations per kernel as warmup\n")

    moe_kernels, total_kernels = calculate_moe_runtimes(csv_file)

    if not moe_kernels:
        print(f"No kernels with 'moe' in name found in {csv_file}")
        print(f"Total kernels processed: {total_kernels}")
        return

    # Group kernels by name
    grouped_kernels = group_kernels_by_name(moe_kernels)

    # print_detailed_results(moe_kernels)

    # Print grouped statistics
    print_grouped_statistics(grouped_kernels, num_warmup)

    # Print comparison table
    print_comparison(grouped_kernels, num_warmup)

    # Overall summary
    print("\n\nOVERALL SUMMARY:")
    print("-" * 40)
    print(f"Total kernels processed:    {total_kernels}")
    print(f"MOE kernels found:          {len(moe_kernels)}")
    print(f"Unique MOE kernel types:    {len(grouped_kernels)}")
    print(f"MOE percentage of total:    {len(moe_kernels)/total_kernels*100:.1f}%")

if __name__ == "__main__":
    main()

