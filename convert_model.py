import tensorflow as tf
print("Loading model...")
model = tf.keras.models.load_model('modelo.keras')
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Writing TFLite model...")
with open('modelo.tflite', 'wb') as f:
      f.write(tflite_model)
  print("Conversion successful!")
