import kagglehub

# Download latest version
path = kagglehub.dataset_download("schmoyote/coffee-reviews-dataset")

print("Path to dataset files:", path)