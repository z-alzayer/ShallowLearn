import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def generate_probability_mesh(classifier, df, feature1, feature2, steps=0.1):
    """
    Generates a mesh grid for the specified features and predicts class probabilities.

    Parameters:
    classifier: Trained classifier model.
    df: DataFrame containing the features.
    feature1: Name of the first feature.
    feature2: Name of the second feature.
    steps: Resolution of the mesh grid.

    Returns:
    xx, yy: Mesh grid coordinates.
    prob_mesh: Predicted class probabilities for each point in the mesh grid.
    """
    x_min, x_max = df[feature1].min() - 1, df[feature1].max() + 1
    y_min, y_max = df[feature2].min() - 1, df[feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, steps),
                         np.arange(y_min, y_max, steps))

    # Prepare mesh grid points for prediction
    mesh_data = np.c_[xx.ravel(), yy.ravel()]

    # Predict class probabilities
    prob_mesh = classifier.predict_proba(mesh_data)
    prob_mesh = prob_mesh.reshape(xx.shape[0], xx.shape[1], -1)

    return xx, yy, prob_mesh





def plot_class_probabilities(xx, yy, prob_mesh, test_data, test_labels):
    """
    Plots the class probabilities as heatmaps for each class.

    Parameters:
    xx, yy: Mesh grid coordinates.
    prob_mesh: Predicted class probabilities for each point in the mesh grid.
    test_data: Test data points to plot.
    test_labels: Actual labels of the test data.
    """
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    num_classes = prob_mesh.shape[2]

    plt.figure(figsize=(18, 6))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i+1)
        plt.contourf(xx, yy, prob_mesh[:,:,i], alpha=0.8, cmap='coolwarm')
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"Class {i} Probability")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def generate_mesh_and_plot(classifier, df, feature1, feature2, test_data, test_labels, steps=0.1):
    """
    Generates a mesh grid for the specified features and plots the class probabilities.

    Parameters:
    classifier: Trained classifier model.
    df: DataFrame containing the features.
    feature1: Name of the first feature.
    feature2: Name of the second feature.
    test_data: Test data points to plot.
    test_labels: Actual labels of the test data.
    steps: Resolution of the mesh grid.
    """
    xx, yy, prob_mesh = generate_probability_mesh(classifier, df, feature1, feature2, steps)
    plot_class_probabilities(xx, yy, prob_mesh, test_data, test_labels)