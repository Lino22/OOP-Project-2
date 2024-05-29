"""
Rosolino Mangano
CIS 3100
Professor Jairam
05/22/2024
"""

import csv
import random
import math
from random import shuffle

# DataHandler class to handle reading CSV file, splitting data into train/test sets, and separating features and labels
class DataHandler:
    def __init__(self, filepath):
        # Constructor to initialize the filepath
        self.filepath = filepath
        self.data = []

    def read_csv(self):
        # Read data from a CSV file and store it in a list
        with open(self.filepath, 'r') as file:
            csv_reader = csv.reader(file)
            self.header = next(csv_reader)  # Store the header row
            self.data = [row for row in csv_reader]
        print("Header Row:", self.header)  # Print the header row to verify the column names

    def train_test_split(self, test_size=0.2):
        # Shuffle the dataset to ensure randomness
        shuffle(self.data)
        # Determine the split index based on the test size
        split_index = int(len(self.data) * (1 - test_size))
        # Split the dataset into training and testing sets
        self.train_data = self.data[:split_index]
        self.test_data = self.data[split_index:]

    def separate_features_labels(self, target_column):
        # Find the index of the target column
        target_index = self.header.index(target_column)
        # Separate the features and labels for training and testing sets
        self.X_train = [list(map(float, row[:target_index] + row[target_index+1:])) for row in self.train_data]
        self.y_train = [row[target_index] for row in self.train_data]
        self.X_test = [list(map(float, row[:target_index] + row[target_index+1:])) for row in self.test_data]
        self.y_test = [row[target_index] for row in self.test_data]

# NaiveBayesClassifier class to implement Naive Bayes algorithm from scratch
class NaiveBayesClassifier:
    def __init__(self):
        # Initialize dictionaries to store the means, standard deviations,
        # and class probabilities for each class
        self.means = {}
        self.stds = {}
        self.class_probabilities = {}

    def fit(self, X, y):
        # Train the classifier by calculating the class probabilities
        # and the means and standard deviations for each feature
        self._calculate_class_probabilities(y)
        self._calculate_means_stds(X, y)

    def _calculate_class_probabilities(self, y):
        # Calculate the probability of each class based on label frequency
        class_counts = {label: y.count(label) for label in set(y)}
        total_count = len(y)
        self.class_probabilities = {label: count / total_count for label, count in class_counts.items()}

    def _calculate_means_stds(self, X, y):
        # Calculate the mean and standard deviation for each class and each feature
        for label in self.class_probabilities:
            # Extract features for instances of the current class
            label_features = [X[i] for i in range(len(X)) if y[i] == label]
            # Calculate mean and standard deviation for each feature
            self.means[label] = [sum(f) / len(f) for f in zip(*label_features)]
            self.stds[label] = [math.sqrt(sum([(x - mean)**2 for x in f]) / len(f)) for mean, f in zip(self.means[label], zip(*label_features))]

    def predict_single(self, input_features):
        # Predict the class of a single feature set
        probabilities = {}
        for label, _ in self.means.items():
            # Start with the prior probability of the class
            probabilities[label] = self.class_probabilities[label]
            # Multiply by the probability of each feature
            for i, feature in enumerate(input_features):
                probabilities[label] *= self._calculate_probability(feature, self.means[label][i], self.stds[label][i])
        # Return the class with the highest probability
        return max(probabilities, key=probabilities.get)

    def _calculate_probability(self, x, mean, std):
        # Calculate the probability of a feature value with a Gaussian distribution
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        return (1 / (math.sqrt(2*math.pi) * std)) * exponent

    def predict(self, X):
        # Predict a list of feature sets
        return [self.predict_single(features) for features in X]

    def accuracy(self, y_test, y_pred):
        # Calculate the accuracy of the predictions
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
        return correct / len(y_test)

    def classification_report(self, y_true, y_pred):
        # Generate a classification report for the predictions
        unique_labels = set(y_true)
        report = {}
        for label in unique_labels:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] == label)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] == label)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] != label)
            tn = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] != label)

            # Calculate precision, recall, and F1-score for each class
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            accuracy = (tp + tn) / len(y_true)

            report[label] = {
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'Accuracy': accuracy
            }

        return report

# KNNClassifier class to implement K-Nearest Neighbors algorithm from scratch
class KNNClassifier:
    def __init__(self, k=3):
        # Initialize KNN with a specified number of neighbors (k)
        self.k = k

    def fit(self, X_train, y_train):
        # Store the training data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Predict the class labels for the test data
        return [self.predict_single(x) for x in X_test]

    def predict_single(self, input_features):
        # Predict the class label for a single feature set
        distances = []
        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(input_features, self.X_train[i])
            distances.append((distance, self.y_train[i]))
        distances.sort(key=lambda x: x[0])  # Sort by distance
        k_nearest_neighbors = distances[:self.k]  # Get the k nearest neighbors
        labels = [label for _, label in k_nearest_neighbors]
        prediction = max(set(labels), key=labels.count)  # Return the most common label
        return prediction

    def euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    def accuracy(self, y_test, y_pred):
        # Calculate the accuracy of the predictions
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
        return correct / len(y_test)

    def classification_report(self, y_true, y_pred):
        # Generate a classification report for the predictions
        unique_labels = set(y_true)
        report = {}
        for label in unique_labels:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] == label)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] == label)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] != label)
            tn = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] != label)

            # Calculate precision, recall, and F1-score for each class
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            accuracy = (tp + tn) / len(y_true)

            report[label] = {
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'Accuracy': accuracy
            }

        return report

# SVMClassifier class to implement a simple linear Support Vector Machine from scratch
class SVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Initialize SVM with specified parameters
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):
        # Convert labels to -1 or 1
        y_train = [1 if label == 'Good' else -1 for label in y_train]  # Assuming 'Good' and 'Bad' labels
        n_samples, n_features = len(X_train), len(X_train[0])

        # Initialize weights and bias
        self.w = [0.0] * n_features
        self.b = 0.0

        # Perform gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X_train):
                condition = y_train[idx] * (sum(self.w[i] * x_i[i] for i in range(n_features)) + self.b) >= 1
                if condition:
                    self.w = [self.w[i] - self.lr * (2 * self.lambda_param * self.w[i]) for i in range(n_features)]
                else:
                    self.w = [self.w[i] - self.lr * (2 * self.lambda_param * self.w[i] - x_i[i] * y_train[idx]) for i in range(n_features)]
                    self.b -= self.lr * y_train[idx]

    def predict(self, X_test):
        # Predict the class labels for the test data
        return [self.predict_single(x) for x in X_test]

    def predict_single(self, input_features):
        # Predict the class label for a single feature set
        linear_output = sum(self.w[i] * input_features[i] for i in range(len(input_features))) + self.b
        return 'Good' if linear_output >= 0 else 'Bad'

    def accuracy(self, y_test, y_pred):
        # Calculate the accuracy of the predictions
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
        return correct / len(y_test)

    def classification_report(self, y_true, y_pred):
        # Generate a classification report for the predictions
        unique_labels = set(y_true)
        report = {}
        for label in unique_labels:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] == label)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] == label)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == label and y_pred[i] != label)
            tn = sum(1 for i in range(len(y_true)) if y_true[i] != label and y_pred[i] != label)

            # Calculate precision, recall, and F1-score for each class
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            accuracy = (tp + tn) / len(y_true)

            report[label] = {
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'Accuracy': accuracy
            }

        return report

# Function to print the classification report in a readable format
def print_classification_report(report):
    print("Classification Report:")
    for label, metrics in report.items():
        print(f"\nClass {label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

# Function to select and return the classifier based on user input
def select_classifier():
    while True:
        print("\nSelect the classifier:")
        print("1: Naive Bayes")
        print("2: K-Nearest Neighbors")
        print("3: Support Vector Machine")
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            return NaiveBayesClassifier()
        elif choice == '2':
            return KNNClassifier(k=3)  # You can adjust the value of k
        elif choice == '3':
            return SVMClassifier()
        else:
            print("Invalid choice. Please select again.")

# Main function to handle user interaction, data processing, and model training/prediction
def main():

    # Defines the path to the CSV file containing the dataset
    filepath = 'banana_quality.csv'
    
    # Initialize DataHandler
    data_handler = DataHandler(filepath)
    data_handler.read_csv()
    data_handler.train_test_split(test_size=0.2)
    data_handler.separate_features_labels(target_column='Quality')

    while True:
        # User selects which classifier to use
        classifier = select_classifier()

        # Fit the classifier
        classifier.fit(data_handler.X_train, data_handler.y_train)

        # Predict and evaluate
        y_pred = classifier.predict(data_handler.X_test)
        accuracy = classifier.accuracy(data_handler.y_test, y_pred)
        report = classifier.classification_report(data_handler.y_test, y_pred)

        # Print accuracy and classification report
        print(f"\nAccuracy: {accuracy:.4f}")
        print_classification_report(report)

        # Simple User Interface for predictions
        while True:
            user_input = input("\nEnter features separated by commas or 'exit' to quit or 'change' to select a new classifier: ")
            if user_input.lower() == 'exit':
                return
            elif user_input.lower() == 'change':
                break
            try:
                input_features = [float(x) for x in user_input.split(',')]
                prediction = classifier.predict_single(input_features)
                print(f"Prediction: {prediction}")
            except ValueError:
                print("Invalid input. Please enter numerical feature values separated by commas.")

if __name__ == "__main__":
    main()
