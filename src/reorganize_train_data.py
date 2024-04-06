from utils.data import mat_to_list
from utils.reorganize_data_folder import reorganize_data_folder

classes_df = mat_to_list('src/data/cars_annos.mat', 'class_names')
reorganize_data_folder('src/data/cars_train', 'src/data/car_dataset_train.csv', 'image_path', 'class', classes_df)
