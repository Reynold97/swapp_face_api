#!/usr/bin/env python3
"""
Automated Load Test Runner for Face Swap API - Simplified Version
Runs the specific load pattern: 0 → 50 → 100 → 150 → 200 users
Each level maintained for 3 minutes
Tests only the default case with single pair of images
"""

import subprocess
import time
import sys
import json
import os
from datetime import datetime, timedelta
import requests
import signal

class LoadTestRunner:
    def __init__(self, api_host, locust_file="locust_face_swap_simple.py"):
        self.api_host = api_host
        self.locust_file = locust_file
        self.load_levels = [50, 100, 150, 200]  # User counts for each level
        self.level_duration = 180  # 3 minutes in seconds
        self.spawn_rate = 50  # Users per second spawn rate
        self.current_process = None
        self.test_start_time = datetime.now()
        
    def check_api_health(self):
        """Check if the API is accessible before starting the test."""
        try:
            print(f"Checking API health at {self.api_host}...")
            # Try to access the API docs endpoint (common for FastAPI)
            response = requests.get(f"{self.api_host}/docs", timeout=10)
            if response.status_code == 200:
                print("✓ API is accessible")
                return True
            else:
                print(f"⚠ API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ API is not accessible: {e}")
            return False
    
    def run_load_level(self, users, duration):
        """Run a specific load level."""
        print(f"\n{'='*50}")
        print(f"Starting load level: {users} users for {duration//60} minutes")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        cmd = [
            "locust",
            "-f", self.locust_file,
            "--host", self.api_host,
            "--headless",
            "-u", str(users),
            "-r", str(self.spawn_rate),
            "-t", f"{duration}s",
            "--html", f"report_{users}_users_{self.test_start_time.strftime('%Y%m%d_%H%M%S')}.html"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in iter(self.current_process.stdout.readline, ''):
                if line.strip():
                    print(f"[LOCUST] {line.strip()}")
            
            self.current_process.wait()
            
            if self.current_process.returncode == 0:
                print(f"✓ Load level {users} users completed successfully")
            else:
                print(f"✗ Load level {users} users failed with return code: {self.current_process.returncode}")
                
        except Exception as e:
            print(f"✗ Error running load level {users}: {e}")
            
        finally:
            self.current_process = None
    
    def run_full_test(self):
        """Run the complete load test with all levels."""
        print("Face Swap API Load Test - Simplified Version")
        print("===========================================")
        print(f"API Host: {self.api_host}")
        print(f"Start Time: {self.test_start_time}")
        print(f"Test Type: Single pair, default options")
        print(f"Test Pattern: {' → '.join(['0'] + [str(u) for u in self.load_levels])} users")
        print(f"Duration per level: {self.level_duration//60} minutes")
        print(f"Total estimated duration: {len(self.load_levels) * self.level_duration//60} minutes")
        print()
        
        # Check API health first
        if not self.check_api_health():
            print("Cannot proceed with load test. Please ensure the API is running and accessible.")
            return False
        
        # Run each load level
        for i, users in enumerate(self.load_levels, 1):
            print(f"\nPhase {i}/{len(self.load_levels)}")
            
            # Calculate remaining time
            remaining_phases = len(self.load_levels) - i + 1
            estimated_completion = datetime.now() + timedelta(seconds=remaining_phases * self.level_duration)
            print(f"Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
            
            self.run_load_level(users, self.level_duration)
            
            # Short break between levels (except after the last one)
            if i < len(self.load_levels):
                print(f"\nBreak between levels... (30 seconds)")
                time.sleep(30)
        
        print(f"\n{'='*60}")
        print("LOAD TEST COMPLETED")
        print(f"{'='*60}")
        print(f"Start Time: {self.test_start_time}")
        print(f"End Time: {datetime.now()}")
        print(f"Total Duration: {datetime.now() - self.test_start_time}")
        print()
        print("Generated files:")
        print("- HTML reports: report_*_users_*.html")
        print("- CSV stats: load_test_stats_*.csv")
        print("- JSON summary: load_test_summary_*.json")
        
        return True
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\nReceived signal {signum}. Stopping current test...")
        if self.current_process:
            self.current_process.terminate()
            self.current_process.wait()
        print("Test stopped by user.")
        sys.exit(0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_test_runner_simple.py <API_HOST>")
        print("Example: python load_test_runner_simple.py http://localhost:8000")
        print("Example: python load_test_runner_simple.py https://api.yourservice.com")
        sys.exit(1)
    
    api_host = sys.argv[1]
    
    # Validate API host format
    if not api_host.startswith(('http://', 'https://')):
        print("Error: API host must start with http:// or https://")
        sys.exit(1)
    
    # Check if locust is installed
    try:
        subprocess.run(["locust", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Locust is not installed or not in PATH")
        print("Install with: pip install locust")
        sys.exit(1)
    
    # Check if the locust file exists
    locust_file = "locust_face_swap_simple.py"
    if not os.path.exists(locust_file):
        print(f"Error: Locust file '{locust_file}' not found")
        print("Make sure the locust file is in the same directory")
        sys.exit(1)
    
    # Create and run the test
    runner = LoadTestRunner(api_host, locust_file)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, runner.signal_handler)
    signal.signal(signal.SIGTERM, runner.signal_handler)
    
    # Ask for confirmation
    print(f"Ready to start simplified load test against {api_host}")
    print("This will test only the default case with single pair of images")
    print("Test duration: approximately 12 minutes + setup time")
    print()
    print("⚠️  Make sure you have updated the image filenames in the locust file!")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Test cancelled.")
        sys.exit(0)
    
    # Run the test
    success = runner.run_full_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()