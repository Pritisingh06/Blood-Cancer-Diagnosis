
import tensorflow as tf
from src.model import create_model
from src.preprocess import load_datasets
import matplotlib.pyplot as plt
import joblib

# Parameters
img_height = 100
img_width = 100
batch_size = 32
dataset_dir = '/content/drive/My Drive/CANCER'
epochs = 20

# Load Data
train_ds, val_ds = load_datasets(dataset_dir, img_height, img_width, batch_size)
num_classes = len(train_ds.class_names)

# Model
model = create_model(img_height, img_width, num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save Model
joblib.dump(model, 'saved_models/model.pkl')

# Plot Training Metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('outputs/training_history.png')
