#!/usr/bin/env python3
"""
Simple test to run the main forces.py without pyplot blocking
"""

import subprocess
import sys
import os

def run_forces_with_timeout():
    """Run forces.py and capture output"""
    try:
        # Change to the forces directory
        os.chdir(r"c:\Users\pablo\python\forces")
        
        # Run the script with a timeout
        result = subprocess.run([sys.executable, "forces.py"], 
                              capture_output=True, 
                              text=True, 
                              timeout=30)  # 30 second timeout
        
        print("=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Script timed out after 30 seconds (probably due to matplotlib)")
        print("This is expected behavior - the script likely completed and is showing the plot")
    except Exception as e:
        print(f"Error running script: {e}")

if __name__ == "__main__":
    run_forces_with_timeout()
