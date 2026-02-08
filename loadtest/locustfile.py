"""HTTP load test skeleton (Locust).

This is optional and only applies if you run an HTTP API around the pipeline.
By default, this repo uses Streamlit UI; if you expose an API route (e.g. FastAPI),
point Locust at it.

Run:
  locust -f loadtest/locustfile.py --host http://localhost:8000
"""

from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def query(self):
        # Adjust endpoint + payload to match your deployed API.
        self.client.post(
            "/query",
            json={"query": "What is self-attention in transformers?"},
            timeout=120,
        )
