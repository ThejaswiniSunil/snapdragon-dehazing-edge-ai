import tensorflow as tf

model = tf.keras.models.load_model("potatoes.h5", compile=False)
model.export("potato_savedmodel")

print("SavedModel exported to potato_savedmodel/")