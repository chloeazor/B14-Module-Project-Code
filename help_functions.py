import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tqdm
import subprocess
import sys

def install_packages():
    # Install TensorFlow and Graphviz using Mamba
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mamba"])  # Ensure mamba is available
    #subprocess.check_call(["mamba", "install", "-y", "-c", "conda-forge", "tensorflow-cpu", "graphviz"])
    
    # Install Keras Visualizer using pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-visualizer"])



def get_total_params(model):
    return model.count_params()

def evaluate_misidentified_and_similar_scores(model, x_test, y_test, threshold=0.1):
    """
    Evaluates the model on test data and returns a list of misidentified test images and those
    with very similar prediction scores for different labels.
    
    Parameters:
    - model: The trained model.
    - x_test: Test data (images).
    - y_test: True labels for the test data.
    - threshold: The difference between the top two scores to consider scores as 'similar'.
    
    Returns:
    - misidentified_indices: List of indices where the model misidentified the label.
    - similar_score_indices: List of indices where the prediction scores for different labels are very similar.
    """
    predictions = model.predict(x_test)
    misidentified_indices = []
    similar_score_indices = []
    
    for i, pred in enumerate(predictions):
        # Find the predicted label and true label
        predicted_label = np.argmax(pred)
        true_label = y_test[i]
        
        # Check for misidentification
        if predicted_label != true_label:
            misidentified_indices.append(i)
        
        # Check for similar scores (top two scores are close)
        sorted_scores = np.sort(pred)[-2:]  # Get the two highest scores
        if sorted_scores[1] - sorted_scores[0] < threshold:
            similar_score_indices.append(i)
    
    return misidentified_indices, similar_score_indices




def plot_prediction_scores(x_test,y_test,predictions,name):
    misidentified=[151, 247, 259, 321, 340, 381, 448, 495, 582, 619, 691, 717, 720, 740, 844, 947, 951, 956, 1014, 1039, 1112, 1128, 1156, 1178, 1182, 1194, 1226, 1232, 1242, 1247, 1260, 1283, 1289, 1319, 1328, 1364, 1393, 1433, 1500, 1522, 1527, 1530, 1549, 1553, 1609, 1621, 1641, 1678, 1681, 1751, 1754, 1790, 1800, 1828, 1878, 1901, 1941, 2004, 2016, 2024, 2043, 2053, 2098, 2109, 2118, 2135, 2182, 2266, 2272, 2293, 2387, 2408, 2422, 2433, 2454, 2462, 2488, 2582, 2607, 2648, 2654, 2713, 2720, 2730, 2743, 2770, 2877, 2915, 2921, 2939, 2953, 2995]
    not_unique=[321, 674, 1039, 1232, 1319, 1433, 1609, 1641, 2053, 2125, 2185, 2408, 2713, 2769, 2863, 3060, 3533, 3751, 3902, 4065, 4154, 4265, 4289]
    
    # Choose 3 random images from x_test
    random_indices = random.sample(range(len(x_test)), 3)

    # Choose 1 random image from misidentified
    misidentified_index = random.choice(misidentified)

    # Choose 1 random image from not_unique
    not_unique_index = random.choice(not_unique)

    # Combine the selected indices
    selected_indices = random_indices + [misidentified_index, not_unique_index]

    # Create a plot with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    fig.suptitle(f"Prediction Scores and True Labels - {name}", fontsize=16)

    # Plot the first row: the actual test images
    for i in range(5):
        axes[0, i].imshow(x_test[selected_indices[i]], cmap='binary')
        axes[0, i].set_title(f"True label: {y_test[selected_indices[i]]}")
        axes[0, i].axis('off')

    # Plot the second row: histograms of prediction scores for each label
    for i in range(5):
        axes[1, i].bar(np.arange(10), predictions[selected_indices[i]], color='gray')
        axes[1, i].set_title("Prediction Scores")
        axes[1, i].set_xticks(np.arange(10))
        axes[1, i].set_xlabel("Label")
        axes[1, i].set_ylabel("Score")

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig("/files/output/Aufgabe1/prediction_score.pdf")


def read_table_with_progress(filepath, chunksize=10000):
    chunks = []
    total_lines = sum(1 for line in open(filepath))  # Count total lines in the file
    with tqdm.tqdm(total=total_lines, desc="Loading Data") as pbar:
        for chunk in pd.read_table(filepath, chunksize=chunksize):
            chunks.append(chunk)
            pbar.update(chunksize)
    return pd.concat(chunks)

def reset_weights(model):
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))



def plot_loss_over_epoch(hist, name): 
    plt.figure()
    title = f"Training and Test Loss Over Epochs - {name}"
    # Plot the training and validation loss over epochs
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Test Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("/files/output/Aufgabe1/loss_over_epochs.pdf")


def plot_acc_over_epoch(hist, name): 
    plt.figure()
    title = f"Training and Test Accuracy Over Epochs - {name}"
    # Plot the training and validation accuracy over epochs
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("/files/output/Aufgabe2/accuracy_over_epochs.pdf")


def plot_random_train_images(x_train, y_train, name, num_images=5, save_path="/files/output/Aufgabe1/MNIST_Data_Inspection.pdf"):
    """
    Plots num_images random images from the dataset with their labels, along with a title.
    
    Parameters:
    - x_train: Training data (images)
    - y_train: Training labels
    - name: Name to be displayed as the title of the plot
    - num_images: Number of random images to plot
    - save_path: File path to save the plot
    """
    
    # Select random indices for the images
    random_indices = np.random.choice(len(x_train), num_images, replace=False)

    # Create the plot
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    # Add the name at the top
    title = f"MNIST Data Inspection - {name}"
    fig.suptitle(title, fontsize=16)

    # Plot num_images random images with labels
    for i, ax in enumerate(axes):
        ax.imshow(x_train[random_indices[i]], cmap='binary')
        ax.set_title(f"Label: {y_train[random_indices[i]]}")
        ax.axis('off')

    # Save the plot to a file
    plt.savefig(save_path)
    #plt.show()
