import json
import time
import csv
import os
from datetime import datetime
from typing import Dict, List, Any
from locust import HttpUser, task, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import logging

# Configure logging
setup_logging("INFO", None)
logger = logging.getLogger(__name__)

class FaceSwapAPIUser(HttpUser):
    """
    Locust user class for testing the Face Swap API swap_url endpoint.
    Simplified version testing only the default case with one pair of images.
    """
    
    def on_start(self):
        """Called when a user starts. Initialize user-specific data."""
        self.user_id = time.time()  # Simple unique identifier
        logger.info(f"User {self.user_id} started")
        
        # Single pair of test images - UPDATE THESE WITH YOUR ACTUAL FILENAMES
        self.face_filename = "bc78a38b-f5db-44d5-ad0a-3f8d1f96d148-w985nt6rr7.jpg"  # Replace with your source face image
        self.model_filename = "8fd9d6c5-3628-4b25-bb2f-3c26d692e55f-z3yyi62b2k.jpg"  # Replace with your target image
        
        # Default processing options (simplified)
        self.default_options = {
            "mode": "one_to_one",
            "direction": "left_to_right", 
            "use_codeformer": False,
            "codeformer_fidelity": 0.5,
            "background_enhance": True,
            "face_upsample": True,
            "upscale": 2,
            "face_refinement_steps": 1
        }

    @task(1)
    def test_swap_url_default(self):
        """
        Test the /swap_url endpoint with default configuration and single image pair.
        """
        # Prepare request payload with single model file
        payload = {
            "face_filename": self.face_filename,
            "model_filenames": [self.model_filename],  # Single target image
            "options": self.default_options
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer oPbj-lIVYhYOgn1CcrYHxFRohmF6-18s-p0zp7remio"
        }
        
        start_time = time.time()
        
        try:
            with self.client.post(
                "/swap_url",
                json=payload,
                headers=headers,
                catch_response=True,
                name="swap_url_default"
            ) as response:
                
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if response.status_code == 200:
                    # Successful response
                    try:
                        response_data = response.json()
                        if "urls" in response_data and len(response_data["urls"]) > 0:
                            response.success()
                            logger.debug(f"User {self.user_id} - Success: {len(response_data['urls'])} image processed")
                        else:
                            response.failure("Empty URLs in successful response")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON in successful response")
                        
                elif response.status_code == 206:
                    # Partial success (shouldn't happen with single image, but handle it)
                    try:
                        response_data = response.json()
                        if "urls" in response_data:
                            response.success()
                            logger.debug(f"User {self.user_id} - Partial success")
                        else:
                            response.failure("Invalid partial success response format")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON in partial success response")
                        
                elif response.status_code == 400:
                    # Bad request
                    logger.warning(f"User {self.user_id} - Bad request: {response.text}")
                    response.failure(f"Bad request: {response.status_code}")
                    
                elif response.status_code == 500:
                    # Server error
                    logger.error(f"User {self.user_id} - Server error: {response.text}")
                    response.failure(f"Server error: {response.status_code}")
                    
                else:
                    # Unexpected status code
                    logger.error(f"User {self.user_id} - Unexpected status: {response.status_code}")
                    response.failure(f"Unexpected status code: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"User {self.user_id} - Request failed with exception: {str(e)}")
            # The exception will be automatically caught by Locust
            
    def on_stop(self):
        """Called when a user stops."""
        logger.info(f"User {self.user_id} stopped")


class StatsCollector:
    """
    Collects and saves detailed statistics during the load test.
    """
    
    def __init__(self):
        self.stats_history = []
        self.start_time = datetime.now()
        self.stats_file = f"load_test_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.summary_file = f"load_test_summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create CSV file with headers
        with open(self.stats_file, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'user_count', 'total_requests', 'failures', 'failure_rate',
                'avg_response_time', 'min_response_time', 'max_response_time',
                'median_response_time', 'rps', 'total_content_length'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def collect_stats(self, environment):
        """Collect current statistics."""
        stats = environment.stats.total
        current_time = datetime.now()
        
        stat_entry = {
            'timestamp': current_time.isoformat(),
            'user_count': environment.runner.user_count,
            'total_requests': stats.num_requests,
            'failures': stats.num_failures,
            'failure_rate': stats.fail_ratio,
            'avg_response_time': stats.avg_response_time,
            'min_response_time': stats.min_response_time or 0,
            'max_response_time': stats.max_response_time,
            'median_response_time': stats.median_response_time,
            'rps': stats.current_rps,
            'total_content_length': stats.total_content_length
        }
        
        self.stats_history.append(stat_entry)
        
        # Write to CSV file
        with open(self.stats_file, 'a', newline='') as csvfile:
            fieldnames = stat_entry.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(stat_entry)
            
        logger.info(f"Stats collected - Users: {stat_entry['user_count']}, "
                   f"RPS: {stat_entry['rps']:.1f}, "
                   f"Avg Response: {stat_entry['avg_response_time']:.1f}ms, "
                   f"Failures: {stat_entry['failure_rate']:.1%}")
    
    def save_summary(self, environment):
        """Save final test summary."""
        stats = environment.stats.total
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            'test_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'target_endpoint': '/swap_url',
                'test_type': 'single_pair_default_options'
            },
            'final_stats': {
                'total_requests': stats.num_requests,
                'total_failures': stats.num_failures,
                'failure_rate': stats.fail_ratio,
                'avg_response_time_ms': stats.avg_response_time,
                'min_response_time_ms': stats.min_response_time or 0,
                'max_response_time_ms': stats.max_response_time,
                'median_response_time_ms': stats.median_response_time,
                'requests_per_second': stats.total_rps,
                'total_content_length_bytes': stats.total_content_length
            },
            'percentiles': {
                '50th': stats.get_response_time_percentile(0.5),
                '90th': stats.get_response_time_percentile(0.9),
                '95th': stats.get_response_time_percentile(0.95),
                '99th': stats.get_response_time_percentile(0.99)
            }
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Test completed. Summary saved to {self.summary_file}")
        logger.info(f"Detailed stats saved to {self.stats_file}")
        
        # Print final summary
        print("\n" + "="*60)
        print("LOAD TEST SUMMARY")
        print("="*60)
        print(f"Test Type: Single pair, default options")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total Requests: {summary['final_stats']['total_requests']}")
        print(f"Total Failures: {summary['final_stats']['total_failures']}")
        print(f"Failure Rate: {summary['final_stats']['failure_rate']:.1%}")
        print(f"Average Response Time: {summary['final_stats']['avg_response_time_ms']:.1f}ms")
        print(f"Requests per Second: {summary['final_stats']['requests_per_second']:.1f}")
        print(f"95th Percentile: {summary['percentiles']['95th']:.1f}ms")
        print(f"99th Percentile: {summary['percentiles']['99th']:.1f}ms")
        print("="*60)


# Global stats collector instance
stats_collector = StatsCollector()

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    logger.info("Load test started")
    print(f"Starting simplified load test at {datetime.now()}")
    print("Test configuration:")
    print("- Endpoint: /swap_url")
    print("- Test type: Single pair, default options")
    print("- Load pattern: 0 → 50 → 100 → 150 → 200 users")
    print("- Each level: 3 minutes")
    print("- Total duration: ~15 minutes")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    logger.info("Load test completed")
    stats_collector.save_summary(environment)

# Set up periodic stats collection (every 10 seconds)
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize periodic stats collection."""
    if environment.web_ui:
        # Running with web UI, stats will be collected via timer
        def collect_stats_periodically():
            while environment.runner.state in ["spawning", "running"]:
                time.sleep(10)  # Collect stats every 10 seconds
                if environment.runner.state in ["spawning", "running"]:
                    stats_collector.collect_stats(environment)
        
        import threading
        stats_thread = threading.Thread(target=collect_stats_periodically, daemon=True)
        stats_thread.start()


# Configuration for the load test
if __name__ == "__main__":
    print("Face Swap API Load Test - Simplified Configuration")
    print("=================================================")
    print()
    print("This test uses:")
    print("- Single source face image")
    print("- Single target model image") 
    print("- Default processing options")
    print("- one_to_one mode with standard enhancement")
    print()
    print("To run this load test, use one of these commands:")
    print()
    print("1. With Web UI (recommended):")
    print("   locust -f locust_face_swap_simple.py --host=http://your-api-host")
    print()
    print("2. Headless mode with specific load pattern:")
    print("   locust -f locust_face_swap_simple.py --host=http://your-api-host \\")
    print("          --headless -u 200 -r 50 -t 900s")
    print()
    print("Load Pattern Details:")
    print("- Start with 0 users")
    print("- Ramp up by 50 users every 3 minutes")
    print("- Target: 0 → 50 → 100 → 150 → 200 users")
    print("- Each plateau lasts 3 minutes")
    print("- Total test duration: ~15 minutes")
    print()
    print("⚠️  IMPORTANT: Before running, update these filenames in the code:")
    print("   - face_filename: Replace 'test_face.jpg' with your source image")
    print("   - model_filename: Replace 'test_model.jpg' with your target image")