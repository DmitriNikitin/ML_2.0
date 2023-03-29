import requests

# тест для проверки статуса ответа API
def test_predict_status():
    response = requests.get("http://127.0.0.1:8000/predict")
    assert response.status_code == 200
