import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('src/data/cars_annos.mat')

# Print the information on the mat file
print(mat_data)
