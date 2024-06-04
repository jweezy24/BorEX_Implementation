import os
import argparse
import csv
import tqdm
import random
import PIL.Image

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from scipy.stats import multivariate_normal
from smqtk_classifier import ClassifyImage
from xaitk_saliency import GenerateImageClassifierBlackboxSaliency


import numpy as np
import matplotlib.pyplot as plt



def app(
    image_filepath: str,
    # Assuming outputs `nClass` length arrays.
    blackbox_classify: ClassifyImage,
    gen_bb_sal: GenerateImageClassifierBlackboxSaliency,
):
    # Load the image
    ref_image = np.asarray(PIL.Image.open(image_filepath))
    sal_maps = gen_bb_sal(ref_image, blackbox_classify)
    print(f"Saliency maps: {sal_maps.shape}")
    visualize_saliency(ref_image, sal_maps)

    
def visualize_saliency(ref_image: np.ndarray, sal_maps: np.ndarray) -> None:    
    # Visualize the saliency heat-maps
    sub_plot_ind = len(sal_maps) + 1
    plt.figure(figsize=(12, 6))
    plt.subplot(2, sub_plot_ind, 1)
    plt.imshow(ref_image)
    plt.axis('off')
    plt.title('Test Image')

    # Some magic numbers here to get colorbar to be roughly the same height
    # as the plotted image.
    colorbar_kwargs = {
        "fraction": 0.046*(ref_image.shape[0]/ref_image.shape[1]),
        "pad": 0.04,
    }

    for i, class_sal_map in enumerate(sal_maps):
        print(f"Class {i} saliency map range: [{class_sal_map.min()}, {class_sal_map.max()}]")

        # Positive half saliency
        plt.subplot(2, sub_plot_ind, 2+i)
        plt.imshow(ref_image, alpha=0.7)
        plt.imshow(
            np.clip(class_sal_map, 0, 1),
            cmap='jet', alpha=0.3
        )
        plt.clim(0, 1)
        plt.colorbar(**colorbar_kwargs)
        plt.title(f"Class #{i+1} Pos Saliency")
        plt.axis('off')

        # Negative half saliency
        plt.subplot(2, sub_plot_ind, sub_plot_ind+2+i)
        plt.imshow(ref_image, alpha=0.7)
        plt.imshow(
            np.clip(class_sal_map, -1, 0),
            cmap='jet_r', alpha=0.3
        )
        plt.clim(-1, 0)
        plt.colorbar(**colorbar_kwargs)
        plt.title(f"Class #{i+1} Neg Saliency")
        plt.axis('off')
    
    plt.show()

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


  
class test_model (ClassifyImage):
    """ Blackbox model to output the two focus classes. """
    
    def __init__(self,M,transformer):
        self.M = M
        self.transformer = transformer
    
    def get_labels(self):
        self.sal_class_labels = {
            0: "0-2",
            1: "3-9",
            2: "10-19",
            3: "20-29",
            4: "30-39",
            5: "40-49",
            6: "50-59",
            7: "60-69",
            8: "more than 70"}
        self.t_labels = list(self.sal_class_labels.values())
        self.t_keys = list(self.sal_class_labels.keys())
        return self.t_labels
    
    def classify_images(self, image_iter):
        # Input may either be an NDaray, or some arbitrary iterable of NDarray images.
        
        for img in image_iter:
            preped_image = self.transformer(img, return_tensors='pt')
            output = self.M(**preped_image)
            proba = output.logits.softmax(1)
            final_probability = proba[0][proba.argmax(1).item()].item()
            preds = proba.argmax(1)
            pred = preds[0].item()
            lst_translation = proba.detach().numpy().tolist()[0]
            yield dict(zip(self.t_labels, lst_translation))
    
    def get_config(self):
        # Required by a parent class.
        return {}

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
    for i in range(0,window_len):
        for j in range(0,X.shape[0]):
            for k in range(0,X.shape[0]):
                Q.append((j,k,i))
    
    return Q

## 1 to 1 translation from paper (doesn't work)
def algorithm_attempt1(image_paths,max_window_size,M,transformer,N,init_length,kernel,observable_set_size= 500):
    window_sizes = [50,64,78,92,107,121,135,150]
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

            optimal_point = [0,0,0]
            last_improvement = 0

            for c in tqdm.tqdm(range(N+init_length)):
                
                
                # if last_improvement != 0 and abs(EIs.min()-last_improvement) <= 0.0001:
                #     print("Converged")
                #     break 
                
                if c < init_length:
                    i = random.choice(window_sizes)
                    j = random.randint(0,X.shape[0])
                    k = random.randint(0, X.shape[1])

                    p = (j,k,i)
                    
                    while p in samples:

                        i = random.choice(window_sizes)
                        j = random.randint(0,X.shape[0])
                        k = random.randint(0, X.shape[1])

                        p = (j,k,i)
                    

                    
                else:
                    p = optimal_point
                    while p in samples:
                        index = random.randint(0,2)
                        
                        j,k,i= p
                        if index == 0:
                            j+=1
                        elif index == 1:
                            k+=1
                        elif index == 2:
                            i = random.choice(window_sizes)
                        
                        p = (j,k,i)


                samples.append(p)
                X_hat = sample_from_D(X,j,k,i)
                f_X_hat = evaluate_test_model(M,transformer,X_hat,label[0])

                y_i = abs(f_X - f_X_hat)

                local_Y.append(y_i)
                local_X.append(p)
                
                if c < init_length:
                    continue
                
                gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,normalize_y=True).fit(local_X, local_Y)
                
                EIs = []
                
                observable_set = []
                
                ## Generate obserable set
                for i in range(observable_set_size):
                    i = random.choice(window_sizes)
                    j = random.randint(0,X.shape[0])
                    k = random.randint(0, X.shape[1])

                    p = (j,k,i)
                    
                    while p in observable_set or p in samples:

                        i = random.choice(window_sizes)
                        j = random.randint(0,X.shape[0])
                        k = random.randint(0, X.shape[1])

                        p = (j,k,i)
                    observable_set.append(p)
                    
                tmp_Y = np.array(local_Y)
                y_star = tmp_Y.min()
                ms,covs = gpr.predict(observable_set,return_std=True)
                # covs = kernel(observable_set,observable_set)
                for i in range(len(observable_set)):
                    m = ms[i]
                    cov = covs[i]

                    # print(f"Means: {m}")
                    # print(f"Covariance: {cov}")
                    
                    var = multivariate_normal(mean=m , cov=cov)
                    
                    # print(m,cov.flatten())
                    cov = cov.flatten()
                    if cov != 0:
                        ps = (m - y_star)/cov
                    else:
                        ps = 0
                        
                    pdf_eval = var.pdf(ps)
                    cdf_eval = var.cdf(ps)
                    # print(f"PS:{ps}")
                    # print(f"PDF: {var.pdf(ps)}")
                    # print(f"CDF: {var.cdf(ps)}")
                    
                    EI = (m-y_star)*cdf_eval + cov*pdf_eval
                    EIs.append(EI)
                    
                EIs = np.array(EIs)
                
                print(f"EIs Min = {EIs.min()}")    
                print(f"EIs Min = {observable_set[EIs.argmax()]}")  
                
                optimal_point = observable_set[EIs.argmax()]
                
                
                
                last_improvement = EIs.min()
                
            tmp_Y = np.array(local_Y)
            optimal_point = local_X[tmp_Y.argmax()]
            p = optimal_point
            # gpr.predict()
            whole_image = Q[:50176]
            print(optimal_point[2])
            optimal_box = []
            for i in range(224):
                for j in range(224):
                    optimal_box.append((i,j,optimal_point[2]))
            
            m1,std2 = gpr.predict(whole_image,return_std=True)
            m2,std2 = gpr.predict(optimal_box,return_std=True)
            # salency_ratio = m1/m2
            # salency_ratio = X/sample_from_D(X,p[0],p[1],p[2])
            # show_image_from_matrix(X)
            show_image_from_matrix(m2.reshape(224,224))
            show_image_from_matrix(std2.reshape(224,224))

## Using other libraries 
def algorithm_attempt2(image_paths,max_window_size,M,transformer,N,init_length,kernel,observable_set_size= 500):
    from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
    from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack

    gen_slidingwindow = SlidingWindowStack((50, 50), (20, 20), threads=4)
    gen_rise = RISEStack(1000, 8, 0.5, seed=0, threads=4, debiased=False)
    gen_rise_debiased = RISEStack(1000, 8, 0.5, seed=0, threads=4, debiased=True)
    window_sizes = [50,64,78,92,107,121,135,150]
    t_M = test_model(M,transformer)
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
            app(
                image_path,
                t_M,
                gen_rise,
            )
            
            
    pass            

def main():
    parser = argparse.ArgumentParser(description="Image Dataset Loader")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-parser for the 'load' command
    load_parser = subparsers.add_parser('load', help="Load image paths and labels from a root directory")
    load_parser.add_argument('--dataset', required=True, type=str, help="Path to the root directory of the dataset")
    load_parser.add_argument('--labels', required=True, type=str, help="Path to the CSV file with labels")

    args = parser.parse_args()
    
    init_length = 20
    max_window_size = 20 
    N = 200

    if args.command == 'load':
        labels = load_labels(args.labels)
        image_paths = load_images_recursive(args.dataset, labels)
        kernel = 1.0 * Matern(length_scale=12, nu=2.5)
        M,transformer = load_test_model()
        
        #Algorithm 2 currently works. Algorithm 1 fails.
        
        # algorithm_attempt1(image_paths,max_window_size,M,transformer,N,init_length,kernel)
        algorithm_attempt2(image_paths,max_window_size,M,transformer,N,init_length,kernel)  
      
            
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()