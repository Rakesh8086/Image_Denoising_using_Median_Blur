import cv2
from mpi4py import MPI
import numpy as np

# Initializing MPI
comm = MPI.COMM_WORLD #creating channel for communication between processes.
rank = comm.Get_rank() #rank of the calling process within the communicator
size = comm.Get_size() #total number of processes within the communicator 

# Input image is loaded
input_image = 'image1.jpg'  # Load the input image 
image = cv2.imread(input_image)

# Verifying if the image is loaded successfully
if image is None:
    print("Error: Could not load the image.")
    MPI.Finalize()
    exit(1)

# Dividing rows of image equally among MPI processes
rows_per_process = image.shape[0] // size
start_row = rank * rows_per_process
end_row = start_row + rows_per_process

# Function to apply median blur to a portion of the image
def median_blur(image_part, kernel_size):
    return cv2.medianBlur(image_part, kernel_size)

# Process the portion of the image assigned to this MPI process
blurred_image = median_blur(image[start_row:end_row], kernel_size=5)  # 

# Combining processed image parts to the root process
combined_parts = comm.gather(blurred_image, root=0)

# Concatenating processed image parts on the root process
if rank == 0:
    processed_image = np.vstack(combined_parts)
else:
    processed_image = None

# Saving the processed image 
if rank == 0:      # to initiate saving only when rank reaches root process
    output_image = 'denoised_image.jpg'
    cv2.imwrite(output_image, processed_image)
    print(f"Denoised image saved as '{output_image}'")

# Finalize MPI
MPI.Finalize()


