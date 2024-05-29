/*
 * Rosolino Mangano
 * CIS 3100
 * Professor Jairam
 * 05/22/2024
 */

#include <iostream> // For input and output stream operations (cin, cout)
#include <fstream>  // For file handling (ifstream)
#include <sstream>  // For string stream operations (istringstream)
#include <vector>   // For using the vector container
#include <string>   // For using the string class
#include <algorithm>// For algorithms like random_shuffle and max_element
#include <cmath>    // For mathematical functions (sqrt, exp, pow)
#include <map>      // For using the map container
#include <set>      // For using the set container
#include <utility> // For using std::pair
#include <cstdlib> // For exit() and EXIT_FAILURE functions

using namespace std;

// Helper function to split a string by a delimiter
vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// DataHandler class to handle reading CSV file, splitting data into train/test sets, and separating features and labels
class DataHandler {
public:
    vector<vector<float> > features;
    vector<string> labels;

    DataHandler(const string& filename) { // Constructor takes filename
        read_csv(filename);
    }

    void read_csv(const string& filename) {
        ifstream file(filename);
        string line;
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            exit(EXIT_FAILURE);
        }

        // Skip the header row
        if (getline(file, line)) {
            // Verify that the header row matches expected columns
            vector<string> headers = split(line, ',');
            cout << "Headers: ";
            for (size_t i = 0; i < headers.size(); ++i) {
                cout << headers[i] << " ";
            }
            cout << endl;
        }

        while (getline(file, line)) {
            vector<string> tokens = split(line, ',');
            vector<float> row_features;
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                try {
                    row_features.push_back(stof(tokens[i]));
                } catch (const invalid_argument& e) {
                    cerr << "Invalid argument: " << e.what() << " at line: " << line << endl;
                    exit(EXIT_FAILURE);
                } catch (const out_of_range& e) {
                    cerr << "Out of range error: " << e.what() << " at line: " << line << endl;
                    exit(EXIT_FAILURE);
                }
            }
            features.push_back(row_features);
            labels.push_back(tokens.back());
        }
        file.close();
    }
};

// NaiveBayesClassifier class to implement Naive Bayes algorithm from scratch
class NaiveBayesClassifier {
public:
    map<string, vector<float> > means; // Container to hold the means of features for each class
    map<string, vector<float> > stds; // Container to hold the standard deviations of features for each class
    map<string, float> class_probabilities; // Container to hold the probabilities of each class

    // Method to train the classifier
    void fit(const vector<vector<float> >& X, const vector<string>& y) {
        calculate_class_probabilities(y); // Calculate class probabilities
        calculate_means_stds(X, y); // Calculate means and standard deviations
    }

    // Method to calculate the probability of each class based on label frequency
    void calculate_class_probabilities(const vector<string>& y) {
        map<string, int> class_counts;
        for (vector<string>::const_iterator label = y.begin(); label != y.end(); ++label) {
            class_counts[*label]++;
        }
        int total_count = y.size();
        for (map<string, int>::const_iterator pair = class_counts.begin(); pair != class_counts.end(); ++pair) {
            class_probabilities[pair->first] = static_cast<float>(pair->second) / total_count;
        }
    }

    // Method to calculate the mean and standard deviation for each class and each feature
    void calculate_means_stds(const vector<vector<float> >& X, const vector<string>& y) {
        map<string, vector<vector<float> > > label_features;
        for (size_t i = 0; i < y.size(); ++i) {
            label_features[y[i]].push_back(X[i]);
        }
        for (map<string, vector<vector<float> > >::const_iterator pair = label_features.begin(); pair != label_features.end(); ++pair) {
            string label = pair->first;
            const vector<vector<float> > &features = pair->second;
            int n_features = features[0].size();
            vector<float> mean(n_features, 0.0);
            vector<float> std(n_features, 0.0);
            for (vector<vector<float> >::const_iterator feature_set = features.begin(); feature_set != features.end(); ++feature_set) {
                for (int i = 0; i < n_features; ++i) {
                    mean[i] += (*feature_set)[i];
                }
            }
            for (int i = 0; i < n_features; ++i) {
                mean[i] /= features.size();
            }
            for (vector<vector<float> >::const_iterator feature_set = features.begin(); feature_set != features.end(); ++feature_set) {
                for (int i = 0; i < n_features; ++i) {
                    std[i] += pow((*feature_set)[i] - mean[i], 2);
                }
            }
            for (int i = 0; i < n_features; ++i) {
                std[i] = sqrt(std[i] / features.size());
            }
            means[label] = mean;
            stds[label] = std;
        }
    }

    // Method to predict the class of a single feature set
    string predict_single(const vector<float>& input_features) {
        map<string, float> probabilities;
        for (map<string, vector<float> >::const_iterator pair = means.begin(); pair != means.end(); ++pair) {
            string label = pair->first;
            probabilities[label] = class_probabilities[label];
            for (size_t i = 0; i < input_features.size(); ++i) {
                probabilities[label] *= calculate_probability(input_features[i], means[label][i], stds[label][i]);
            }
        }

        // Find the label with the maximum probability
        string max_label;
        float max_prob = -1.0;
        for (map<string, float>::const_iterator prob = probabilities.begin(); prob != probabilities.end(); ++prob) {
            if (prob->second > max_prob) {
                max_prob = prob->second;
                max_label = prob->first;
            }
        }

        return max_label;
    }

    // Method to calculate the probability of a feature value with a Gaussian distribution
    float calculate_probability(float x, float mean, float std) {
        float exponent = exp(-pow(x - mean, 2) / (2 * pow(std, 2)));
        return (1 / (sqrt(2 * M_PI) * std)) * exponent;
    }

    // Method to predict a list of feature sets
    vector<string> predict(const vector<vector<float> >& X) {
        vector<string> predictions;
        for (vector<vector<float> >::const_iterator features = X.begin(); features != X.end(); ++features) {
            predictions.push_back(predict_single(*features));
        }
        return predictions;
    }

    // Method to calculate the accuracy of the predictions
    float accuracy(const vector<string>& y_test, const vector<string>& y_pred) {
        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (y_test[i] == y_pred[i]) {
                correct++;
            }
        }
        return static_cast<float>(correct) / y_test.size();
    }

    // Method to generate a classification report for the predictions
    map<string, map<string, float> > classification_report(const vector<string>& y_true, const vector<string>& y_pred) {
        set<string> unique_labels(y_true.begin(), y_true.end());
        map<string, map<string, float> > report;
        for (set<string>::const_iterator label = unique_labels.begin(); label != unique_labels.end(); ++label) {
            int tp = 0, fp = 0, fn = 0, tn = 0;
            for (size_t i = 0; i < y_true.size(); ++i) {
                if (y_true[i] == *label && y_pred[i] == *label) {
                    tp++;
                } else if (y_true[i] != *label && y_pred[i] == *label) {
                    fp++;
                } else if (y_true[i] == *label && y_pred[i] != *label) {
                    fn++;
                } else {
                    tn++;
                }
            }
            float precision = (tp + fp) == 0 ? 0 : tp / static_cast<float>(tp + fp);
            float recall = (tp + fn) == 0 ? 0 : tp / static_cast<float>(tp + fn);
            float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
            float accuracy = (tp + tn) / static_cast<float>(y_true.size());
            map<string, float> metrics;
            metrics["Precision"] = precision;
            metrics["Recall"] = recall;
            metrics["F1-score"] = f1;
            metrics["Accuracy"] = accuracy;
            report[*label] = metrics;
        }
        return report;
    }
};

// KNNClassifier class to implement K-Nearest Neighbors algorithm from scratch
class KNNClassifier {
private:
    int k;  // Number of nearest neighbors to consider
    vector<vector<float> > X_train;  // Training features
    vector<string> y_train;  // Training labels

public:
    // Constructor to initialize the number of neighbors
    explicit KNNClassifier(int k_val = 3) : k(k_val) {}

    // Store the training data
    void fit(const vector<vector<float> >& X, const vector<string>& y) {
        X_train = X;
        y_train = y;
    }

    // Predict the class of a single feature set
    string predict_single(const vector<float>& input_features) {
        vector<pair<float, string> > distances; // Store distances and corresponding labels

        // Calculate Euclidean distance from input features to all training samples
        for (size_t i = 0; i < X_train.size(); ++i) {
            float distance = 0.0f;
            for (size_t j = 0; j < X_train[i].size(); ++j) {
                distance += pow(X_train[i][j] - input_features[j], 2);
            }
            distance = sqrt(distance); // Square root of sum of squares
            distances.push_back(make_pair(distance, y_train[i])); // Store distance and label
        }

        // Sort distances
        sort(distances.begin(), distances.end());

        // Count the labels among the k nearest neighbors
        map<string, int> label_count;
        for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
            label_count[distances[i].second]++;
        }

        // Determine the most frequent label
        string prediction;
        int max_count = -1; // Initialize to -1 to ensure any count will be higher initially
        for (map<string, int>::const_iterator label = label_count.begin(); label != label_count.end(); ++label) {
            if (label->second > max_count) {
                max_count = label->second;
                prediction = label->first;
            }
        }

        return prediction; // Return the most common label
    }

    // Predict a list of feature sets
    vector<string> predict(const vector<vector<float> >& X) {
        vector<string> predictions;
        for (vector<vector<float> >::const_iterator features = X.begin(); features != X.end(); ++features) {
            predictions.push_back(predict_single(*features));
        }
        return predictions;
    }

    // Method to calculate the accuracy of the predictions
    float accuracy(const vector<string>& y_test, const vector<string>& y_pred) {
        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (y_test[i] == y_pred[i]) {
                correct++;
            }
        }
        return static_cast<float>(correct) / y_test.size();
    }

    // Classification report generation
    map<string, map<string, float> > classification_report(const vector<string>& y_true, const vector<string>& y_pred) {
        map<string, map<string, float> > report;
        map<string, int> true_counts, pred_counts, true_positive;

        for (size_t i = 0; i < y_true.size(); ++i) {
            true_counts[y_true[i]]++;
            pred_counts[y_pred[i]]++;
            if (y_true[i] == y_pred[i]) {
                true_positive[y_true[i]]++;
            }
        }

        for (map<string, int>::const_iterator label = true_counts.begin(); label != true_counts.end(); ++label) {
            float precision = (pred_counts[label->first] == 0) ? 0 : true_positive[label->first] / static_cast<float>(pred_counts[label->first]);
            float recall = true_positive[label->first] / static_cast<float>(label->second);
            float f1 = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);
            float accuracy = static_cast<float>(true_positive[label->first]) / y_true.size();

            map<string, float> metrics;
            metrics["Precision"] = precision;
            metrics["Recall"] = recall;
            metrics["F1-score"] = f1;
            metrics["Accuracy"] = accuracy;

            report[label->first] = metrics;
        }

        return report;
    }
};

// Support Vector Machine Classifier
class SVMClassifier {
private:
    vector<float> weights;
    float bias;
    float learning_rate;
    float lambda_param;
    int n_iters;

public:
    // Constructor to initialize SVM parameters
    SVMClassifier(float lr = 0.001, float lambda = 0.01, int n = 1000) : learning_rate(lr), lambda_param(lambda), n_iters(n), bias(0.0) {}

    // Method to train the SVM classifier
    void fit(const vector<vector<float> >& X, const vector<string>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.resize(n_features, 0.0);

        vector<int> y_binary(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            y_binary[i] = (y[i] == "Good") ? 1 : -1;
        }

        for (int _ = 0; _ < n_iters; ++_) {
            for (int i = 0; i < n_samples; ++i) {
                float condition = y_binary[i] * (dot_product(X[i], weights) + bias);
                if (condition >= 1) {
                    for (int j = 0; j < n_features; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j]);
                    }
                } else {
                    for (int j = 0; j < n_features; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j] - X[i][j] * y_binary[i]);
                    }
                    bias -= learning_rate * y_binary[i];
                }
            }
        }
    }

    // Method to predict the class of a single feature set
    string predict_single(const vector<float>& input_features) {
        float linear_output = dot_product(input_features, weights) + bias;
        return (linear_output >= 0) ? "Good" : "Bad";
    }

    // Method to predict a list of feature sets
    vector<string> predict(const vector<vector<float> >& X) {
        vector<string> predictions;
        for (vector<vector<float> >::const_iterator features = X.begin(); features != X.end(); ++features) {
            predictions.push_back(predict_single(*features));
        }
        return predictions;
    }

    // Method to calculate the accuracy of the predictions
    float accuracy(const vector<string>& y_test, const vector<string>& y_pred) {
        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (y_test[i] == y_pred[i]) {
                correct++;
            }
        }
        return static_cast<float>(correct) / y_test.size();
    }

    // Classification report generation
    map<string, map<string, float> > classification_report(const vector<string>& y_true, const vector<string>& y_pred) {
        set<string> unique_labels(y_true.begin(), y_true.end());
        map<string, map<string, float> > report;
        for (set<string>::const_iterator label = unique_labels.begin(); label != unique_labels.end(); ++label) {
            int tp = 0, fp = 0, fn = 0, tn = 0;
            for (size_t i = 0; i < y_true.size(); ++i) {
                if (y_true[i] == *label && y_pred[i] == *label) {
                    tp++;
                } else if (y_true[i] != *label && y_pred[i] == *label) {
                    fp++;
                } else if (y_true[i] == *label && y_pred[i] != *label) {
                    fn++;
                } else {
                    tn++;
                }
            }
            float precision = (tp + fp) == 0 ? 0 : tp / static_cast<float>(tp + fp);
            float recall = (tp + fn) == 0 ? 0 : tp / static_cast<float>(tp + fn);
            float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
            float accuracy = (tp + tn) / static_cast<float>(y_true.size());
            map<string, float> metrics;
            metrics["Precision"] = precision;
            metrics["Recall"] = recall;
            metrics["F1-score"] = f1;
            metrics["Accuracy"] = accuracy;
            report[*label] = metrics;
        }
        return report;
    }

private:
    // Helper function to calculate dot product
    float dot_product(const vector<float>& a, const vector<float>& b) {
        float result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
};

// Function to print the classification report in a readable format
void print_classification_report(const map<string, map<string, float> >& report) {
    cout << "Classification Report:\n";
    for (map<string, map<string, float> >::const_iterator label_metrics = report.begin(); label_metrics != report.end(); ++label_metrics) {
        cout << "\nClass " << label_metrics->first << ":\n";
        for (map<string, float>::const_iterator metric = label_metrics->second.begin(); metric != label_metrics->second.end(); ++metric) {
            cout << "  " << metric->first << ": " << metric->second << "\n";
        }
    }
}

// Main function to handle user interaction, data processing, and model training/prediction
int main() {
    
    // Initialize DataHandler with a specific CSV file path
    DataHandler data("banana_quality.csv");

    // Classifier instances
    NaiveBayesClassifier nb;
    KNNClassifier knn(3);
    SVMClassifier svm;

    // Fit the classifiers with the training data
    nb.fit(data.features, data.labels);
    knn.fit(data.features, data.labels);
    svm.fit(data.features, data.labels);

    // Predict using the test data
    vector<string> nb_predictions = nb.predict(data.features);
    vector<string> knn_predictions = knn.predict(data.features);
    vector<string> svm_predictions = svm.predict(data.features);

    // Generate the classification reports
    map<string, map<string, float> > nb_report = nb.classification_report(data.labels, nb_predictions);
    map<string, map<string, float> > knn_report = knn.classification_report(data.labels, knn_predictions);
    map<string, map<string, float> > svm_report = svm.classification_report(data.labels, svm_predictions);

    // Print the classification reports
    cout << "Naive Bayes Classifier Report:\n";
    print_classification_report(nb_report);

    cout << "\nK-Nearest Neighbors Classifier Report:\n";
    print_classification_report(knn_report);

    cout << "\nSupport Vector Machine Classifier Report:\n";
    print_classification_report(svm_report);

    return 0;
}
