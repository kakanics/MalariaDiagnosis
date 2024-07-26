import tensorflow_datasets as tfds

# Define the builder for the malaria dataset with the data_dir
builder = tfds.builder('malaria', data_dir='dataset')

# Download and prepare the dataset
builder.download_and_prepare()

# Print the data directory
print(f"Dataset saved in: {builder.data_dir}")

# Load the dataset from the local directory
dataset = builder.as_dataset()

# Get dataset information
info = builder.info

# Print dataset information
print(info)