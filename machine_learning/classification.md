# Classification, KNN, and Logistic Regression

## Introduction to Classification

Classification is a fundamental task in **Supervised Machine Learning**. The goal of classification is to train a model using a dataset of labeled examples (where we know the correct category for each example) so that the model can accurately predict the category (or "class") for new, unseen data points.

Think of it like sorting objects into predefined bins. You learn the rules by looking at examples that are already sorted, and then you use those rules to sort new objects.

**Key Characteristics:**

*   **Supervised:** Requires labeled training data (input features and their corresponding correct class labels).
*   **Discrete Output:** The predicted output is a category or class label, not a continuous number (like predicting house prices, which is Regression).
*   **Goal:** To learn a decision boundary that separates different classes in the feature space.

**Types of Classification:**

1.  **Binary Classification:** Problems where there are only two possible outcomes or classes.
    *   *Example:* Spam detection (email is either "spam" or "not spam").
    *   *Example:* Medical test result (patient has the "disease" or "does not have the disease").
2.  **Multiclass Classification:** Problems where there are more than two possible outcomes or classes.
    *   *Example:* Image classification (image could be a "cat", "dog", "car", or "person").
    *   *Example:* Sentiment analysis (text could be "positive", "negative", or "neutral").

**How is Classification Applied in the Real World?**

Classification algorithms are used across various domains:

*   **Email Filtering:** Classifying emails as spam or not spam.
*   **Medical Diagnosis:** Predicting whether a patient has a certain disease based on symptoms or medical imaging (e.g., classifying tumors as malignant or benign).
*   **Image Recognition:** Identifying objects within images (e.g., autonomous driving systems identifying pedestrians, traffic lights, other vehicles).
*   **Sentiment Analysis:** Determining the sentiment expressed in a piece of text (e.g., positive, negative, neutral review).
*   **Document Classification:** Assigning topics or categories to documents (e.g., news articles classified as "sports", "politics", "technology").

---

##  K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive classification algorithms. The core idea is based on the principle: **"You are similar to your neighbors."**

Imagine you have a map with different houses plotted on it, each colored based on who lives there (e.g., blue for Democrats, red for Republicans). Now, a new house is built. To predict the political leaning of the new resident, KNN would look at the 'K' closest existing houses (neighbors) to the new one. If the majority of the K closest neighbors are, say, blue, KNN would predict that the new resident is also likely to be a Democrat.

**How KNN Works :**

1.  **Choose 'K':** Decide how many neighbors (K) to consider. This is a crucial parameter.
2.  **Calculate Distances:** When a new, unclassified data point arrives, calculate the distance between this new point and *every* point in the training dataset. Common distance measures include Euclidean distance (the straight-line distance).
3.  **Identify Neighbors:** Find the 'K' training data points that are closest (have the smallest distances) to the new point. These are its "K nearest neighbors".
4.  **Majority Vote:** Look at the class labels of these K neighbors. The new data point is assigned the class label that is most common among its K nearest neighbors. (In case of a tie, strategies like picking randomly, or preferring the absolute closest neighbor might be used).

**Key notes:**

*   **Lazy Learner:** KNN is called a "lazy learner" because it doesn't build an explicit model during the training phase. It simply stores the entire training dataset. The real computation happens during the prediction phase.
*   **Instance-Based:** It makes predictions based on the similarity to specific instances in the training data.
*   **Importance of K:** Choosing the right value for K is critical. A small K might be sensitive to noise, while a large K might smooth out the decision boundary too much.
*   **Feature Scaling:** KNN is sensitive to the scale of features (e.g., if one feature ranges from 0-1 and another from 0-1000, the second feature will dominate the distance calculation). It's usually essential to scale features (e.g., to a 0-1 range or standardize them) before applying KNN.

**Simple Python Implementation (scikit-learn):**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris # A common dataset for classification examples
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # To scale features

#  Load Data (Example: Iris dataset)
# Iris dataset has 3 classes of iris plants based on 4 features (sepal/petal length/width)
iris = load_iris()
X = iris.data  # Features
y = iris.target # Labels 

#  Split Data into Training and Testing sets
# test_size=0.3 means 30% of data is for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#  Feature Scaling (Important for KNN!)
# Scale data so that features have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on training data

#  Create and Train the KNN Model
# Choose a value for K (n_neighbors)
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the model (for KNN, this mostly involves storing the data)
knn_classifier.fit(X_train_scaled, y_train)

#  Make Predictions on the Test Set
y_pred = knn_classifier.predict(X_test_scaled)

#  Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN (K={k}) Model Accuracy: {accuracy:.4f}")

```

---

##  Logistic Regression

Despite its name containing "Regression", Logistic Regression is a widely used algorithm for **Classification** problems, especially binary classification.

Instead of directly predicting a class label, Logistic Regression models the **probability** that a given input point belongs to a particular class.

Think of it as trying to draw a line (or a curve in higher dimensions) that best separates the different classes. For a new data point, the algorithm calculates its position relative to this line and uses that to estimate the probability of it belonging to the class on one side versus the other.

**How Logistic Regression Works (Abstract Steps):**

1.  **Linear Combination:** Just like Linear Regression, it starts by calculating a weighted sum of the input features, plus a bias term. `z = w1*x1 + w2*x2 + ... + wn*xn + b`
2.  **Sigmoid Function:** The result `z` can be any real number. To convert this into a probability (which must be between 0 and 1), the result `z` is passed through a special mathematical function called the **Sigmoid function** (or Logistic function).
    *   The Sigmoid function takes any real number and squashes it into the range (0, 1).
    *   Large positive values of `z` result in a probability close to 1.
    *   Large negative values of `z` result in a probability close to 0.
    *   A value of `z = 0` results in a probability of 0.5.
3.  **Probability Output:** The output of the Sigmoid function is interpreted as the probability of the data point belonging to the "positive" class (usually denoted as class 1). For example, P(Class=1). The probability of belonging to the "negative" class (Class=0) is simply 1 - P(Class=1).
4.  **Decision Threshold:** A threshold (typically 0.5) is used to make the final classification decision.
    *   If the predicted probability P(Class=1) is greater than the threshold (e.g., > 0.5), the model predicts Class 1.
    *   If the predicted probability is less than or equal to the threshold (e.g., <= 0.5), the model predicts Class 0.

**Key Points:**

*   **Probability Estimates:** Provides probabilities, which can be useful for understanding confidence or for ranking predictions.
*   **Linear Decision Boundary:** In its basic form, Logistic Regression finds a linear decision boundary. It works well when the classes are reasonably separable by a line or hyperplane. (It can be extended for non-linear boundaries using techniques like polynomial features or kernels, but the basic concept is linear).
*   **Interpretability:** The weights (coefficients) learned by the model can often be interpreted to understand the influence of each feature on the prediction outcome (specifically, on the log-odds of the outcome).
*   **Training:** Unlike KNN's lazy approach, Logistic Regression involves an active training process where the algorithm iteratively adjusts the weights (`w`) and bias (`b`) to minimize prediction errors on the training data (often using techniques like Gradient Descent).

**Simple Python Implementation (scikit-learn):**

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris # Can use the same Iris dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

#  Load Data (Using Iris dataset again for consistency)
iris = load_iris()
X = iris.data
y = iris.target

#  Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#  Feature Scaling (Generally recommended for Logistic Regression, helps convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Create and Train the Logistic Regression Model
# 'max_iter' might need adjustment depending on the dataset for convergence
log_reg_classifier = LogisticRegression(random_state=42, max_iter=200)

# Train the model
log_reg_classifier.fit(X_train_scaled, y_train)

#  Make Predictions on the Test Set
y_pred_log = log_reg_classifier.predict(X_test_scaled)

#  Evaluate the Model
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Model Accuracy: {accuracy_log:.4f}")

# Display more detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names))
```
