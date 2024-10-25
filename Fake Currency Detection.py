from google.colab import drive

drive.mount('/content/drive')
folder_path = '/content/drive/My Drive/Indian Currency Dataset'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil
from PIL import Image
import numpy as np
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train = os.path.join(folder_path, 'train')
test = os.path.join(folder_path, 'test')

train_fake = os.path.join(train, 'fake')
train_real = os.path.join(train, 'real')

train_fake_images = [os.path.join(train_fake, img) for img in os.listdir(train_fake)]
train_real_images = [os.path.join(train_real, img) for img in os.listdir(train_real)]

train_fake, val_fake = train_test_split(train_fake_images, test_size=0.15, random_state=42)
train_real, val_real = train_test_split(train_real_images, test_size=0.15, random_state=42)

train_images = train_fake + train_real
val_images = val_fake + val_real

validation = os.path.join(folder_path, 'validation')
os.makedirs(validation, exist_ok=True)

val_fake = os.path.join(validation, 'fake')
val_real = os.path.join(validation, 'real')
os.makedirs(val_fake, exist_ok=True)
os.makedirs(val_real, exist_ok=True)

for img_path in val_fake:
    shutil.move(img_path, os.path.join(val_fake, os.path.basename(img_path)))

for img_path in val_real:
    shutil.move(img_path, os.path.join(val_real, os.path.basename(img_path)))

print(f'Train set size: {len(train_images)}')
print(f'Validation set size: {len(val_images)}')

batch_size = 32
target_size = (224, 224)
train_generator = datagen.flow_from_directory(
    train,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=42
)

validation_generator = datagen.flow_from_directory(
    validation,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    seed=42
)

test_generator = datagen.flow_from_directory(
    test,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    seed=42
)

print(os.listdir(train))
print(os.listdir(validation))
print(os.listdir(test))

model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
model.trainable = False

flatten_layer = tf.keras.layers.Flatten()(model.output)
dropout_layer = tf.keras.layers.Dropout(0.5)(flatten_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)

model = tf.keras.models.Model(model.input, output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

num_epochs = 5
model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}')

predictions = model.predict(test_generator)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_currency(image_path, model):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] >= 0.5:
       return "Real Currency"
    else:
       return "Fake Currency"

image_path = '/content/drive/My Drive/Indian Currency Dataset/2000.jpg'
result = predict_currency(image_path, model)
print(f"The predicted result is: {result}")

image_path = '/content/drive/My Drive/Indian Currency Dataset/test (52).jpg'
result = predict_currency(image_path, model)
print(f"The predicted result is: {result}")

