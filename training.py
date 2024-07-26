import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)

# Determine the size of the dataset
dataset_size = info.splits['train'].num_examples

# Define the split sizes
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split the dataset
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Define a function to resize images
def resize_image(image, label):
    image = tf.image.resize(image, [128, 128])
    return image, label

# Apply the resize function to the datasets
train_dataset = train_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print(f'Validation accuracy: {accuracy:.2f}')

# Save the model
model.save("MalariaModel.keras")