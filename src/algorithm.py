import os
import argparse
import csv
import tqdm
import random

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt


def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions

def load_labels(csv_path):
    """Load labels from FairFace's CSV file."""
    labels = {}
    with open(csv_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            image_path, age, gender, ethnicity, _ = row
            labels[image_path] = (age,gender,ethnicity)
    return labels

def save_image_as_jpg(image_matrix, file_path):
    """
    Saves an image matrix as a JPG file.
    
    Parameters:
    - image_matrix: numpy array of shape (224, 224, 3), the image in matrix form.
    - file_path: str, the path where the JPG file will be saved.
    """
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_matrix.astype('uint8'), 'RGB')
    
    # Save the image as a JPG file
    image.save(file_path, 'JPEG')

def grey_out_box(image_matrix, start_i, start_j, window_size):
    """
    Greys out a box in the image starting at (start_i, start_j) with the specified width and height.
    
    Parameters:
    - image_matrix: numpy array of shape (224, 224, 3), the image in matrix form.
    - start_i: int, the starting pixel row for the box.
    - start_j: int, the starting pixel column for the box.
    - width: int, the width of the box.
    - height: int, the height of the box.
    
    Returns:
    - The modified image with the greyscale box.
    """
    # Create a copy of the image to modify
    modified_image = np.copy(image_matrix)
    
    # Define the grey color (128, 128, 128)
    grey_color = [128, 128, 128]
    
    # Calculate the end points of the box
    end_i = min(start_i + window_size, 224)
    end_j = min(start_j + window_size, 224)
    
    # Grey out the specified box
    modified_image[start_i:end_i, start_j:end_j, :] = grey_color
    
    
    return modified_image




def load_images_recursive(root_dir, labels):
    """Recursively load all image paths from the root directory and their labels."""
    image_paths = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if is_image_file(filename):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, root_dir)
                label = labels.get(relative_path)
                print(full_path,label)
                image_paths[full_path] = label
    return image_paths

def load_image_as_matrix(image_path):
    """Load an image from the given path into a matrix (numpy array)."""
    try:
        with Image.open(image_path) as img:
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def sample_from_D(X : np.ndarray, window_size,j,k ):
    
    #Generate grey box for image
    gen_image = grey_out_box(X,j,k,window_size)
    
    return gen_image

def show_image_from_matrix(image_matrix):
    """Display an image from its matrix form using matplotlib."""
    if image_matrix is not None:
        plt.imshow(image_matrix)
        plt.axis('off')  # Hide axes for better display
        plt.show()
    else:
        print("No image data to display.")

def load_test_model():
    """Load a pre-trained gender classification model from TensorFlow Hub."""
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    return model,transforms

def evaluate_test_model(M,transformer,X,label):
    labels = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "more than 70"}

    preped_image = transformer(X, return_tensors='pt')
    output = M(**preped_image)
    proba = output.logits.softmax(1)
    final_probability = proba[0][proba.argmax(1).item()].item()
    preds = proba.argmax(1)
    pred = preds[0].item()

    
    return final_probability


def create_Q(X,window_len):
    Q = []
    for i in range(1,window_len):
        for j in range(0,X.shape[0]):
            for k in range(0,X.shape[0]):
                Q.append((j,k,i))
    
    return Q

def main():
    parser = argparse.ArgumentParser(description="Image Dataset Loader")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-parser for the 'load' command
    load_parser = subparsers.add_parser('load', help="Load image paths and labels from a root directory")
    load_parser.add_argument('--dataset', required=True, type=str, help="Path to the root directory of the dataset")
    load_parser.add_argument('--labels', required=True, type=str, help="Path to the CSV file with labels")

    args = parser.parse_args()
    
    init_length = 100
    max_window_size = 200 + init_length
    N = 1000

    if args.command == 'load':
        labels = load_labels(args.labels)
        image_paths = load_images_recursive(args.dataset, labels)
        kernel = 1.0 * Matern(length_scale=12, nu=2.5)
        M,transformer = load_test_model()
        
        for data in image_paths.items():
            samples = []
            print(data)
            image_path = data[0]
            label = data[1]
            if label == None:
                continue
            
            X = load_image_as_matrix(image_path)
            Q = create_Q(X,max_window_size)
            
            f_X = evaluate_test_model(M,transformer,X,label[0])
            local_X = []
            local_Y = []

            

            for c in tqdm.tqdm(range(N)):
                

                i = random.randint(1,max_window_size)
                j = random.randint(0,X.shape[0])
                k = random.randint(0, X.shape[1])

                p = (j,k,i)

                while p in samples:

                    i = random.randint(1,max_window_size)
                    j = random.randint(0,X.shape[0])
                    k = random.randint(0, X.shape[1])

                    p = (j,k,i)

                samples.append(p)
                X_hat = sample_from_D(X,i,j,k)
                f_X_hat = evaluate_test_model(M,transformer,X_hat,label[0])

                y_i = f_X - f_X_hat

                local_Y.append(y_i)
                local_X.append(p)
                
                
                gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(local_X, local_Y)

                A = random.sample(Q,1000)
            
                m,cov = gpr.predict(A,return_cov=True)
                
                var = multivariate_normal(mean=m , cov=cov)
                print(cov.shape)
                
                                # Create a grid of points
                x1 = np.linspace(0, 6, 50)
                x2 = np.linspace(0, 6, 50)
                X1, X2 = np.meshgrid(x1, x2)
                grid_points = np.c_[X1.ravel(), X2.ravel()]


                # Ensure the covariance matrix is positive definite
                cov += 1e-10 * np.eye(cov.shape[0])

                # Evaluate the PDF on the grid
                pdf_values = np.zeros(X1.shape)

                for i in range(len(grid_points)):
                    # For each grid point, use the corresponding mean and the variance (diagonal of the covariance matrix)
                    mean_i = m[i]
                    cov_ii = cov[i, i]  # Variance of the i-th prediction
                    print(multivariate_normal.pdf(grid_points[i], mean=mean_i, cov=cov_ii))
                    pdf_values.ravel()[i] = multivariate_normal.pdf(grid_points[i], mean=mean_i, cov=cov_ii)

                # Plot the PDF
                plt.figure(figsize=(10, 8))
                plt.contourf(X1, X2, pdf_values.reshape(X1.shape), levels=50, cmap='viridis')
                plt.colorbar(label='PDF')
                plt.scatter(local_X[:, 0], local_X[:, 1], c='red', marker='x', label='Training Points')
                plt.xlabel('X1')
                plt.ylabel('X2')
                plt.title('PDF of Gaussian Process Predictions')
                plt.legend()
                plt.show()

                # print(m,cov)
                

    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()