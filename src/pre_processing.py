from utils.data_processing import load_data, preprocess_data

if __name__ == "__main__":
    # Define data directory
    train_data_dir = 'data/cars_train/'
    test_data_dir = 'data/cars_test/'

    # Load and pre-process training data
    train_dataset = load_data(train_data_dir)
    train_loader = preprocess_data(train_dataset, image_size=(100, 100), batch_size=64)

    # Load and pre-process test data
    test_dataset = load_data(test_data_dir)
    test_loader = preprocess_data(test_dataset, image_size=(100, 100), batch_size=64)

    # Example usage: Iterate through the train_loader
    for images, labels in train_loader:
        # Your training loop code here
        pass
