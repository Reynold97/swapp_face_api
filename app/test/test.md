# Face Swap API Load Test Setup Guide - Simplified Version

## Overview

This simplified load test focuses on testing the default case of your Face Swap API with a single pair of images (one source face, one target model) using default processing options.

## Prerequisites

1. **Python 3.7+** installed
2. **Locust** load testing framework
3. **Access to your Face Swap API**
4. **Two test images** in your GCP bucket (source face + target model)

## Installation

### 1. Install Required Packages
```bash
pip install locust requests
```

### 2. Download the Test Files
Save these files in the same directory:
- `locust_face_swap_simple.py` (main Locust test file)
- `load_test_runner_simple.py` (automated runner script)

## Configuration

### Update Test Images
Before running the test, you **must** update the image filenames in `locust_face_swap_simple.py`:

```python
# Lines ~20-22 in locust_face_swap_simple.py
self.face_filename = "your_source_face.jpg"    # Replace with your source face image
self.model_filename = "your_target_model.jpg"  # Replace with your target image
```

**Example:**
```python
self.face_filename = "person_a_face.jpg"
self.model_filename = "person_b_body.jpg"
```

### Default Test Configuration
The test uses these default settings:
- **Mode**: `one_to_one` (standard single face swap)
- **CodeFormer**: `false` (uses standard enhancement)
- **Upscale**: `2x`
- **Background enhance**: `true`
- **Face upsample**: `true`
- **Face refinement steps**: `1`

## Running the Load Test

### Option 1: Automated Runner (Recommended)
This runs the exact pattern you requested: 0→50→100→150→200 users, 3 minutes each level.

```bash
python load_test_runner_simple.py http://your-api-host:8000
```

**Examples:**
```bash
python load_test_runner_simple.py http://localhost:8000
python load_test_runner_simple.py https://api.yourservice.com
```

### Option 2: Manual Locust Execution

#### With Web UI:
```bash
locust -f locust_face_swap_simple.py --host=http://your-api-host:8000
```
Then open http://localhost:8089 in your browser.

#### Headless Mode:
```bash
locust -f locust_face_swap_simple.py \
       --host=http://your-api-host:8000 \
       --headless -u 200 -r 50 -t 900s \
       --html report.html
```

## Load Test Pattern

The test follows this exact pattern:
- **Phase 1**: 0 → 50 users (3 minutes)
- **Phase 2**: 50 → 100 users (3 minutes)  
- **Phase 3**: 100 → 150 users (3 minutes)
- **Phase 4**: 150 → 200 users (3 minutes)

**Total Duration**: ~12 minutes + setup time

Each user repeatedly calls the `/swap_url` endpoint with the same image pair and default options.

## Generated Reports

After the test completes, you'll find these files:

### 1. CSV Statistics (`load_test_stats_YYYYMMDD_HHMMSS.csv`)
Time-series data collected every 10 seconds with columns:
- `timestamp`: When the measurement was taken
- `user_count`: Number of active users at this time
- `total_requests`: Cumulative request count
- `failures`: Number of failed requests
- `failure_rate`: Percentage of failed requests (0.0 to 1.0)
- `avg_response_time`: Average response time in milliseconds
- `min_response_time`: Minimum response time in milliseconds
- `max_response_time`: Maximum response time in milliseconds
- `median_response_time`: Median response time in milliseconds
- `rps`: Current requests per second
- `total_content_length`: Total bytes transferred

### 2. JSON Summary (`load_test_summary_YYYYMMDD_HHMMSS.json`)
Final test summary containing:
```json
{
  "test_info": {
    "start_time": "2024-01-15T10:30:00",
    "end_time": "2024-01-15T10:42:30", 
    "duration_seconds": 750,
    "test_type": "single_pair_default_options"
  },
  "final_stats": {
    "total_requests": 1250,
    "total_failures": 12,
    "failure_rate": 0.0096,
    "avg_response_time_ms": 2340.5,
    "requests_per_second": 8.7
  },
  "percentiles": {
    "50th": 2100.0,
    "90th": 4200.0,
    "95th": 5100.0,
    "99th": 7800.0
  }
}
```

### 3. HTML Reports (`report_*_users_YYYYMMDD_HHMMSS.html`)
Visual reports with graphs and charts for each load level (when using automated runner).

## Key Metrics to Monitor

### 1. Response Time
- **Average**: Should remain reasonable under load (< 10 seconds ideally)
- **95th percentile**: Indicates worst-case user experience
- **Trend**: Watch for degradation as user count increases

### 2. Failure Rate
- **Target**: Should remain very low (< 1%)
- **Types of failures**: 
  - 400 errors: Usually invalid image files or API configuration
  - 500 errors: Server overload or crashes

### 3. Requests per Second (RPS)
- **Throughput**: Shows actual API capacity
- **Saturation point**: Where RPS stops increasing despite more users

### 4. Expected Performance Baseline
For face swapping APIs, typical performance ranges:
- **Response time**: 2-10 seconds per request (GPU dependent)
- **Throughput**: 5-50 RPS (depending on hardware)
- **Memory usage**: High during processing (watch for memory leaks)

## Troubleshooting

### Common Issues:

1. **"Connection refused"**:
   ```
   Solution: Verify API is running on the specified host and port
   Check: curl http://your-host:8000/docs
   ```

2. **"Failed to download image" errors**:
   ```
   Solution: Verify your image filenames exist in the GCP bucket
   Check: Update face_filename and model_filename in the test file
   ```

3. **High failure rates (>5%)**:
   ```
   Possible causes:
   - API server overloaded
   - Images too large/complex
   - GPU memory exhaustion
   - Database connection issues
   ```

4. **Very slow response times (>30 seconds)**:
   ```
   Possible causes:
   - Insufficient GPU resources
   - CPU bottlenecks
   - Network latency to GCP bucket
   - Memory swapping
   ```

### Debugging Steps:

1. **Start small**: Test with 1-5 users first
   ```bash
   locust -f locust_face_swap_simple.py --host=http://your-host -u 5 -r 1 -t 60s
   ```

2. **Monitor server resources** during test:
   - GPU utilization (`nvidia-smi`)
   - CPU usage (`top` or `htop`)
   - Memory usage
   - Disk I/O

3. **Check API logs** for error patterns

4. **Manual verification**:
   ```bash
   curl -X POST http://your-host:8000/swap_url \
        -H "Content-Type: application/json" \
        -d '{
          "face_filename": "your_source_face.jpg",
          "model_filenames": ["your_target_model.jpg"],
          "options": {
            "mode": "one_to_one",
            "use_codeformer": false,
            "upscale": 2
          }
        }'
   ```

## Performance Expectations

### Typical Results by Load Level:

- **50 users**: Response time 2-5 seconds, minimal failures
- **100 users**: Response time 3-8 seconds, <1% failures  
- **150 users**: Response time 5-12 seconds, <2% failures
- **200 users**: Response time 8-20 seconds, may see higher failure rates

### Red Flags:
- Response times > 30 seconds consistently
- Failure rates > 5%
- RPS decreasing as users increase
- Memory usage constantly growing

## Customization

### Modify Load Pattern
Edit `load_test_runner_simple.py`:
```python
self.load_levels = [25, 50, 75, 100]  # Different user counts
self.level_duration = 300  # 5 minutes per level
self.spawn_rate = 25  # Slower ramp-up
```

### Test Different Options
Edit the default options in `locust_face_swap_simple.py`:
```python
self.default_options = {
    "mode": "one_to_one",
    "use_codeformer": True,  # Test with CodeFormer
    "upscale": 4,  # More intensive processing
    "face_refinement_steps": 3  # Multiple refinement passes
}
```

### Add Request Validation
Modify the response handling to check specific response content:
```python
if response.status_code == 200:
    response_data = response.json()
    if len(response_data.get("urls", [])) != 1:
        response.failure("Expected exactly 1 URL in response")
```

## Best Practices

1. **Baseline first**: Run single-user test to establish baseline performance
2. **Monitor continuously**: Watch server metrics during the test
3. **Document environment**: Record server specs, image sizes, etc.
4. **Coordinate testing**: Ensure you have permission to load test
5. **Use realistic data**: Test with actual image sizes you'll use in production

## Safety Notes

- **GPU intensive**: Face swapping uses significant GPU resources
- **Memory usage**: Monitor for memory leaks during extended tests
- **Start conservatively**: Begin with lower user counts
- **Have monitoring**: Watch server health continuously
- **Test environment**: Use staging/test environment when possible

## Quick Start Checklist

- [ ] Install locust: `pip install locust requests`
- [ ] Update image filenames in `locust_face_swap_simple.py`
- [ ] Verify API is running: `curl http://your-host:8000/docs`
- [ ] Test manually first with single request
- [ ] Run load test: `python load_test_runner_simple.py http://your-host:8000`
- [ ] Monitor server resources during test
- [ ] Analyze results in generated CSV/JSON files