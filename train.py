import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

input_data, labels = load_galaxy_data()
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

data_generator = ImageDataGenerator(rescale=1./255)
training_iterator = data_generator.flow(x_train, y_train, batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

model.fit(
  training_iterator, 
  steps_per_epoch=len(x_train)/5,
  epochs=12,
  validation_data=validation_iterator,
  validation_steps=len(x_valid)/5
)
