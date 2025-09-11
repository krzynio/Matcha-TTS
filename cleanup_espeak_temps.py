#!/usr/bin/env python3
"""
Cleanup script to remove temporary libespeak-ng.so files created by multiprocessing.
Run this script to clean up the /tmp directory from accumulated espeak libraries.
"""

import os
import glob
import shutil
from pathlib import Path

def cleanup_espeak_temps():
    """Remove temporary directories containing libespeak-ng.so files"""
    print("Cleaning up temporary espeak directories...")
    
    # Find all temporary directories with libespeak files
    temp_dirs = []
    for temp_dir in glob.glob("/tmp/tmp*"):
        if os.path.isdir(temp_dir):
            espeak_file = os.path.join(temp_dir, "libespeak-ng.so.1.1.51")
            if os.path.exists(espeak_file):
                temp_dirs.append(temp_dir)
    
    print(f"Found {len(temp_dirs)} temporary directories with espeak files")
    
    # Remove the directories
    removed_count = 0
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
            removed_count += 1
            print(f"Removed: {temp_dir}")
        except Exception as e:
            print(f"Failed to remove {temp_dir}: {e}")
    
    print(f"Successfully removed {removed_count} temporary directories")
    
    # Check remaining count
    remaining_dirs = []
    for temp_dir in glob.glob("/tmp/tmp*"):
        if os.path.isdir(temp_dir):
            espeak_file = os.path.join(temp_dir, "libespeak-ng.so.1.1.51")
            if os.path.exists(espeak_file):
                remaining_dirs.append(temp_dir)
    
    print(f"Remaining temporary espeak directories: {len(remaining_dirs)}")
    
    if remaining_dirs:
        print("Note: Some directories may still be in use by running processes.")
        print("They should be cleaned up automatically when those processes exit.")

if __name__ == "__main__":
    cleanup_espeak_temps()
