import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# # Generate a synthetic dataset with 10 classes, some of which may be colinear
# X, y = make_blobs(n_samples=500, centers=10, random_state=42, cluster_std=1.6)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Create and train the logistic regression model
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(X_train, y_train)

# # Create a mesh to plot the decision boundaries
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))

# # Predict the class for each point in the mesh
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot the decision boundaries
# plt.figure(figsize=(12, 8))
# plt.contourf(xx, yy, Z, alpha=0.3)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.nipy_spectral)
# plt.title("Logistic Regression Decision Boundaries (10 Classes)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# Generate a synthetic dataset with 3-4 classes, some of which may be colinear
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Create a mesh to plot the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class probabilities for each point in the mesh
Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_proba = Z_proba.reshape(xx.shape[0], xx.shape[1], -1)

# Plot the probability distribution for each class
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(4):
    ax = axs[i // 2, i % 2]
    proba_plot = ax.contourf(xx, yy, Z_proba[:,:,i], alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.nipy_spectral)
    ax.set_title(f"Class {i} Probability")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(proba_plot, ax=ax)

plt.tight_layout()
plt.show()


# from sklearn.cluster import KMeans

# # Generate a synthetic dataset (unsupervised, without predefined labels)
# n_classes = 6
# X, _ = make_blobs(n_samples=300, centers=n_classes, random_state=42, cluster_std=1.5)

# # Apply KMeans clustering to label the data unsupervised
# kmeans = KMeans(n_clusters=n_classes, random_state=0)
# unsupervised_labels = kmeans.fit_predict(X)

# # Split the dataset into training and testing sets based on unsupervised labels
# X_train, X_test, y_train, y_test = train_test_split(X, unsupervised_labels, test_size=0.2, random_state=0)

# # Create and train the logistic regression model on the unsupervised labels
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(X_train, y_train)


# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))

# # Predict the class probabilities for each point in the mesh
# Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
# Z_proba = Z_proba.reshape(xx.shape[0], xx.shape[1], -1)

# # Plot the probability distribution for each unsupervised class
# fig, axs = plt.subplots(3, 2, figsize=(15, 10))
# for i in range(n_classes):
#     ax = axs[i // 2, i % 2]
#     proba_plot = ax.contourf(xx, yy, Z_proba[:,:,i], alpha=0.8)
#     ax.scatter(X[:, 0], X[:, 1], c=unsupervised_labels, edgecolors='k', marker='o', cmap=plt.cm.nipy_spectral)
#     ax.set_title(f"Unsupervised Class {i} Probability")
#     ax.set_xlabel("Feature 1")
#     ax.set_ylabel("Feature 2")
#     fig.colorbar(proba_plot, ax=ax)

# plt.tight_layout()
# plt.show()

# #%%

# import xgboost as xgb
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from matplotlib.colors import ListedColormap
# from sklearn.ensemble import RandomForestClassifier
# # Re-create the dataset and train the model (as the previous environment was reset)

# # Create a synthetic dataset
# X, y = make_classification(n_samples=100, n_features=2, n_classes=3, n_informative=2, n_redundant=0, 
#                            n_clusters_per_class=1, random_state=42)

# # Splitting dataset into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Train an XGBoost Classifier
# xgb_model = RandomForestClassifier()
# xgb_model.fit(X_train, y_train)



# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))

# # Predict class probabilities for each point in the mesh with XGBoost model
# xgb_prob_mesh = xgb_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
# xgb_prob_mesh = xgb_prob_mesh.reshape(xx.shape[0], xx.shape[1], -1)


# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# # Plotting heatmaps for each class using XGBoost probabilities
# plt.figure(figsize=(18, 6))

# for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.contourf(xx, yy, xgb_prob_mesh[:,:,i], alpha=0.8, cmap='coolwarm')
#     plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title(f"XGBoost Class {i} Probability")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.colorbar()

# plt.tight_layout()
# plt.show()

# %%
