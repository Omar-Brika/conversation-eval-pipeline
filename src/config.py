class Config:
    def __init__(self, model_name: str, port: int, test_samples: str, verbose: bool):
        self.model_name = model_name
        self.test_samples = test_samples
        self.verbose = verbose
        self.llm_config = {
            "base_url": f"http://localhost:{port}/v1",
            "api_key": "notneeded",
            "model": model_name,
            "temperature": 0.0,
            "max_retries": 2,
            "max_tokens": 512,
        }
