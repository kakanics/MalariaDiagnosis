import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load the model
model = tf.keras.models.load_model("MalariaModel.keras")

# Save the model architecture to a file
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print("Model architecture saved to model.png")