import requests

# тест для проверки возвращаемых результатов модели
def test_predict_result():
    response = requests.get("http://127.0.0.1:8000/predict")
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert response.json()["prediction"] >= 0 and response.json()["prediction"] <= 9
