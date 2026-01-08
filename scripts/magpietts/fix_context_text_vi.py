#!/usr/bin/env python3
# fix_vietnamese_text.py
"""
Fix Vietnamese Text in Lhotse Cuts

This script fixes the text field in Vietnamese Lhotse cuts by:
1. Ensuring text starts with a space
2. Capitalizing the first letter (after the space)
3. Adding period "." at the end if missing

Input: Various Vietnamese cuts directories
Output: Fixed cuts with corrected text formatting
"""

import glob
import logging
import os
import re
from functools import partial

from lhotse import CutSet
from rich import print
from tqdm import tqdm


def fix_text_field(text):
    """
    Fix the text field by:
    1. Normalize to have exactly ONE space at start
    2. Capitalize the first letter
    3. Add period "." at the end if missing
    """
    if not text:
        return text
    
    # Strip all spaces
    text = text.strip()
    
    if not text:
        return text
    
    # Capitalize first letter
    text = text[0].upper() + text[1:]
    
    # Add period if no punctuation
    if not text[-1] in '.!?':
        text = text + "."
    
    # Always add ONE space at start
    return " " + text

def fix_text_in_cut(cut):
    """
    Fix the text field in a cut.
    
    Args:
        cut: A Lhotse Cut object
    
    Returns:
        cut: The modified Cut object with updated text
        changes: Dict with old and new values for logging
    """
    changes = {
        'text_old': None,
        'text_new': None,
        'changed': False,
    }
    
    # Fix text field
    if cut.supervisions[0].text:
        old_text = cut.supervisions[0].text
        changes['text_old'] = old_text
        new_text = fix_text_field(old_text)
        cut.supervisions[0].text = new_text
        changes['text_new'] = new_text
        changes['changed'] = (old_text != new_text)
    
    return cut, changes


def batch_replace_and_write(cut_filepath, new_cut_filepath, show_changes=False):
    """
    Process a single Lhotse shard file by fixing text.
    
    Args:
        cut_filepath (str): Path to the input shard file
        new_cut_filepath (str): Path where the modified shard file will be saved
        show_changes (bool): If True, print changes for first few samples
    
    Returns:
        stats: Dict with processing statistics
    """
    cuts = CutSet.from_file(cut_filepath)
    
    cuts_fixed = []
    all_changes = []
    stats = {'total': 0, 'changed': 0, 'unchanged': 0}
    
    for cut in cuts:
        fixed_cut, changes = fix_text_in_cut(cut)
        cuts_fixed.append(fixed_cut)
        all_changes.append(changes)
        
        stats['total'] += 1
        if changes['changed']:
            stats['changed'] += 1
        else:
            stats['unchanged'] += 1
    
    # Save to file
    CutSet.from_cuts(cuts_fixed).to_file(new_cut_filepath)
    
    # Show changes for first few samples if requested
    if show_changes:
        print("\n    Sample changes (first 3 that changed):")
        shown = 0
        for i, changes in enumerate(all_changes):
            if changes['changed'] and shown < 3:
                print(f"\n    Sample {i+1}:")
                print(f"      BEFORE: '{changes['text_old']}'")
                print(f"      AFTER:  '{changes['text_new']}'")
                shown += 1
    
    return stats


def find_and_verify_shards(cuts_dir: str):
    """
    Find and validate all Lhotse shard files in the specified directory.
    
    Args:
        cuts_dir (str): Directory path containing the shard files
    
    Returns:
        list[str]: Sorted list of paths to all shard files
    """
    cuts_shard_pattern = os.path.join(cuts_dir, "cuts.*.jsonl.gz")
    all_cuts_shard_paths = sorted(glob.glob(cuts_shard_pattern))
    
    if not all_cuts_shard_paths:
        msg = f"No input cut shards found matching pattern: {cuts_shard_pattern}. Cannot proceed."
        logging.error(msg)
        raise FileNotFoundError(msg)
    
    num_total_shards = len(all_cuts_shard_paths)
    
    # Verify shard indices are contiguous and start from 0
    first_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[0]).group(1)
    last_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[-1]).group(1)
    first_idx = int(first_idx_str)
    last_idx = int(last_idx_str)
    expected_last_idx = num_total_shards - 1
    
    if first_idx != 0:
        raise ValueError(f"Expected first shard index to be 0, but found {first_idx}")
    if last_idx != expected_last_idx:
        raise ValueError(f"Expected last shard index to be {expected_last_idx}, but found {last_idx}")
    
    logging.info(f"Verified {num_total_shards} total shard files, with indices from {first_idx} to {last_idx}.")
    return all_cuts_shard_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Vietnamese text formatting')
    parser.add_argument('--output-base-dir', type=str, 
                       default='/data/lhotse/vi',
                       help='Base directory for output')
    args = parser.parse_args()
    
    # Define the Vietnamese datasets to process
    # Input from convai (read-only), output to edgeai (writable)
    datasets = [
        # From YAML - nvyt datasets
        {
            "name": "nvyt_yt12k",
            "input_path": "/vietnamese/Vi_nvyt/nvyt_yt12k/cuts",
            "output_path": f"{args.output_base_dir}/nvyt_yt12k/cuts",
        },
        {
            "name": "nvyt_yt2025",
            "input_path": "/vietnamese/Vi_nvyt/nvyt_yt2025/cuts",
            "output_path": f"{args.output_base_dir}/nvyt_yt2025/cuts",
        },
        # Original datasets
        # {
        #     "name": "vi_infore_1_2_lsvsc",
        #     "input_path": "/data/lhotse/vi/vi_infore_1_2_lsvsc/cuts",
        #     "output_path": f"{args.output_base_dir}/vi_infore_1_2_lsvsc/cuts_fixed",
        # },
        # {
        #     "name": "vi_Long",
        #     "input_path": "/data/lhotse/vi/vi_Long/cuts",
        #     "output_path": f"{args.output_base_dir}/vi_Long/cuts_fixed",
        # },
        # {
        #     "name": "vi_Long_contextText_ipa",
        #     "input_path": "/data/lhotse/vi/vi_Long_contextText/cuts_ipa",
        #     "output_path": f"{args.output_base_dir}/vi_Long_contextText/cuts_ipa_fixed",
        # },
        # {
        #     "name": "vi_Phung_cuts",
        #     "input_path": "/data/lhotse/vi/vi_Phung/lhotse_shar/cuts",
        #     "output_path": f"{args.output_base_dir}/vi_Phung/lhotse_shar/cuts_fixed",
        # },
        # {
        #     "name": "vi_Phung_cuts_textContext",
        #     "input_path": "/data/lhotse/vi/vi_Phung/lhotse_shar/cuts_textContext",
        #     "output_path": f"{args.output_base_dir}/vi_Phung/lhotse_shar/cuts_textContext_fixed",
        # },
    ]
    
    overall_stats = {'total': 0, 'changed': 0, 'unchanged': 0}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset['name']}")
        print(f"Input:  {dataset['input_path']}")
        print(f"Output: {dataset['output_path']}")
        print(f"{'='*80}\n")
        
        input_dir = dataset['input_path']
        output_dir = dataset['output_path']
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"⚠️  WARNING: Input directory does not exist, skipping: {input_dir}\n")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all shards
        try:
            all_cuts_shard_paths = find_and_verify_shards(input_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️  WARNING: {e}, skipping dataset\n")
            continue
        
        # Process each shard
        dataset_stats = {'total': 0, 'changed': 0, 'unchanged': 0}
        
        for idx, cut_filepath in enumerate(tqdm(all_cuts_shard_paths, total=len(all_cuts_shard_paths), desc=f"Processing {dataset['name']}")):
            cut_basename = os.path.basename(cut_filepath)
            cut_filepath_fixed = os.path.join(output_dir, cut_basename)
            show_changes = (idx == 0)  # Show changes for first shard only
            
            stats = batch_replace_and_write(cut_filepath, cut_filepath_fixed, show_changes=show_changes)
            
            dataset_stats['total'] += stats['total']
            dataset_stats['changed'] += stats['changed']
            dataset_stats['unchanged'] += stats['unchanged']
        
        # Print dataset statistics
        print(f"\n📊 Statistics for {dataset['name']}:")
        print(f"  Total samples: {dataset_stats['total']}")
        print(f"  Changed: {dataset_stats['changed']} ({dataset_stats['changed']/dataset_stats['total']*100:.1f}%)")
        print(f"  Unchanged: {dataset_stats['unchanged']} ({dataset_stats['unchanged']/dataset_stats['total']*100:.1f}%)")
        
        # Update overall stats
        overall_stats['total'] += dataset_stats['total']
        overall_stats['changed'] += dataset_stats['changed']
        overall_stats['unchanged'] += dataset_stats['unchanged']
        
        # Validate by showing comparison from the last shard
        print(f"\n{'='*80}")
        print(f"Validation - BEFORE/AFTER comparison:")
        print(f"{'='*80}\n")
        
        # Load original and fixed cuts
        cuts_original = CutSet.from_file(cut_filepath)
        cuts_fixed = CutSet.from_file(cut_filepath_fixed)
        
        shown = 0
        for cut_orig, cut_fixed in zip(cuts_original, cuts_fixed):
            text_orig = cut_orig.supervisions[0].text
            text_fixed = cut_fixed.supervisions[0].text
            
            if text_orig != text_fixed and shown < 3:
                print(f"Example {shown+1}:")
                print(f"  BEFORE: '{text_orig}'")
                print(f"  AFTER:  '{text_fixed}'")
                print()
                shown += 1
            
            if shown >= 3:
                break
    
    print(f"\n{'='*80}")
    print("✅ DONE! Overall Statistics:")
    print(f"{'='*80}")
    print(f"  Total samples processed: {overall_stats['total']}")
    print(f"  Changed: {overall_stats['changed']} ({overall_stats['changed']/overall_stats['total']*100:.1f}%)")
    print(f"  Unchanged: {overall_stats['unchanged']} ({overall_stats['unchanged']/overall_stats['total']*100:.1f}%)")
    print(f"\n📁 New cuts saved to: {args.output_base_dir}")
    print(f"{'='*80}\n")