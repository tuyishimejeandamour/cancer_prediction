from locust import HttpUser, task, between

class SkinCancerUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        # We need a sample image to send. 
        # In a real scenario, we'd have a test image available.
        # Here we will try to use a dummy file if it exists, or skip.
        try:
            with open("test_image.jpg", "rb") as f:
                self.client.post("/predict", files={"file": f})
        except FileNotFoundError:
            # If no file, just hit the root endpoint to simulate load
            self.client.get("/")

    @task(1)
    def index(self):
        self.client.get("/")
