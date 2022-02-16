# Use Mnist dataset to train a teset_model
# You can use own model or parameters
# Remember to check the path you save model

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# about my model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train and save model
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test,  y_test, verbose=2)
model.save('C:/Users/s1551/Desktop/tf_test/test_model.h5')

# check original model
reload_model = tf.keras.models.load_model('C:/Users/s1551/Desktop/tf_test/test_model.h5')
reload_model.summary()
print("\n\n")

# convert .h5 to .tflite
model = tf.keras.models.load_model('C:/Users/s1551/Desktop/tf_test/test_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('C:/Users/s1551/Desktop/tf_test/converted_model.tflite', "wb").write(tflite_model)

# Get the info. of .tflite
interpreter = tf.lite.Interpreter(model_path='C:/Users/s1551/Desktop/tf_test/converted_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(str(input_details))
print("\n\n")
output_details = interpreter.get_output_details()
print(str(output_details))