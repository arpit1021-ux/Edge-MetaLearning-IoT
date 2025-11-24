import requests
import numpy as np

def send_to_server(window):
    r = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"window": window.tolist()}
    )
    print("Server Response:", r.json())

# Example
if __name__ == "__main__":
    fake_window = np.random.randn(6,128)
    send_to_server(fake_window)
