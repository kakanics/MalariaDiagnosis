import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model("MalariaModel.keras")

# Load the dataset using TensorFlow Datasets
dataset, info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)

# Define a function to resize images
def resize_image(image, label):
	image = tf.image.resize(image, [128, 128])
	return image, label

# Apply the resize function to the dataset
val_dataset = dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch the dataset for efficient evaluation
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Function to visualize predictions
def visualize_predictions(dataset, model, num_images=16, save_path='predictions.png'):
	plt.figure(figsize=(10, 10))
	for images, labels in dataset.take(1):
		predictions = model.predict(images)
		for i in range(num_images):
			ax = plt.subplot(4, 4, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(f"True: {int(labels[i])}, Pred: {int(predictions[i] > 0.5)}")
			plt.axis("off")
	plt.savefig(save_path)
	print(f"Predictions saved to {save_path}")

# Visualize predictions on the validation dataset and save to PNG
visualize_predictions(val_dataset, model)