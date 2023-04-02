import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import FastAPI

# загрузка данных
mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()


x_test = x_test / 255.0

# загрузка сохраненной модели
my_model = keras.models.load_model('my_model.h5')

# создание экземпляра FastAPI
app = FastAPI()

# путь для  API
@app.get("/predict")
async def predict():
    # использование модели для предсказания
    predictions = my_model.predict(x_test)
    return {"Значение первого элемента в тестовых данных": int(np.argmax(predictions[0]))}

# запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
