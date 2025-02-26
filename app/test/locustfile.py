from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):

    @task
    def swap_url2(self):
        # Define headers with Authorization token
        headers = {
            "Authorization": "Bearer ..."
        }
        
        # Define the query parameters
        params = {
            "face_filename": "!1.jpg"
        }
        
        # Define the list of model filenames to send in the body
        payload = ["!2.png"]
        
        # Send the POST request to the /swap_url2 endpoint
        with self.client.post("/swap_url2", params=params, json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "urls" in data:
                    # Mark the request as successful if the response contains the 'urls' field
                    response.success()
                else:
                    # Mark the request as failed if the 'urls' field is missing
                    response.failure("No 'urls' field in response")
            else:
                # Mark the request as failed if the status code is not 200
                response.failure(f"Failed with status code {response.status_code}")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)
    host = "..."  # Updated base URL


