import requests

# URL of the MLflow model server
url = "http://127.0.0.1:5002/invocations"

# Example input data (replace with your columns and data)
data = {
    "columns": ["Age", "MonthlyCharges", "Contract"],
    "data": [[35, 70, 1]]
}

# Send POST request
response = requests.post(url, json=data)

# Print model predictions
print(response.json())
