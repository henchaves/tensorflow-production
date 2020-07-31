from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
for i in range(5):
    Image.fromarray(X_test[i]).save(f"uploads/{i}.png")