import json
import requests
import numpy as np

def test_predict():
    url = "http://localhost:8000/predict"
    # Загружаем тестовые данные
    x_test = np.random.rand(1, 28, 28)
    # Посылаем POST-запрос на сервер
    response = requests.post(url, data=json.dumps(x_test.tolist()))
    # Проверяем код ответа
    assert response.status_code == 200
    # Проверяем, что ответ содержит ключ "Значение первого элемента в тестовых данных"
    assert "Значение первого элемента в тестовых данных" in response.json()
    # Проверяем, что предсказанное значение является целым числом
    assert isinstance(response.json()["Значение первого элемента в тестовых данных"], int)
