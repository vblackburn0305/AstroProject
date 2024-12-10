# Imports
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# File path
file_path = 'skyserver.csv'

# Data Initialization
data = {cls: {feature: [] for feature in ["u", "g", "r", "i", "z", "red"]} for cls in ["STAR", "GALAXY", "QSO"]}

# Read Data
try:
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            obj_type, features = row[0], list(map(float, row[1:]))
            if obj_type in data:
                for i, feature in enumerate(["u", "g", "r", "i", "z", "red"]):
                    data[obj_type][feature].append(features[i])
except Exception as e:
    print(f"Error reading file: {e}")

# Compute Stats Function
def compute_stats(data_dict):
    return {
        feature: {"mean": np.mean(values), "std": np.std(values)}
        for feature, values in data_dict.items()
    }

# Compute Stats
stats = {cls: compute_stats(features) for cls, features in data.items()}

# Print Stats
header = f"{'':<10}{'u':<20}{'g':<20}{'r':<20}{'i':<20}{'z':<20}{'redshift':<20}"
print(header)
print("-" * len(header))
for cls, cls_stats in stats.items():
    row = f"|{cls:<10}"
    for feature in ["u", "g", "r", "i", "z", "red"]:
        row += f"{cls_stats[feature]['mean']:.5f}±{cls_stats[feature]['std']:.5f} | "
    print(row)

# Plot Histograms
for cls, color in zip(["STAR", "GALAXY", "QSO"], ['blue', 'green', 'red']):
    sns.histplot(data[cls]["r"], color=color, label=cls, kde=True, alpha=0.6)

plt.xlabel('R-band Magnitude')
plt.ylabel('Frequency')
plt.title('Distribution of R-band Magnitude')
plt.legend(title='Object Type')
plt.show()

# Histogram Analysis Function
def analyze_histogram(data, bins=30, range=(0, 30)):
    mu, sigma = norm.fit(data)
    bin_values, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    gauss_fit = lambda x, mu, sigma, a: a * norm.pdf(x, loc=mu, scale=sigma)
    pars, cov = curve_fit(gauss_fit, centers, bin_values, p0=[mu, sigma, max(bin_values)])
    print(f"μ={pars[0]:.5f}, σ={pars[1]:.5f}, μ_err={np.sqrt(cov[0, 0]):.5f}, σ_err={np.sqrt(cov[1, 1]):.5f}")

# Analyze Histograms
for cls in ["STAR", "GALAXY", "QSO"]:
    analyze_histogram(data[cls]["r"])

# Confusion Matrix
classes = ["STAR", "GALAXY", "QSO"]
conf_matrix = np.zeros((3, 3))
for i, cls1 in enumerate(classes):
    for j, cls2 in enumerate(classes):
        within_1_sigma = np.logical_and(
            np.array(data[cls2]["r"]) >= stats[cls1]["r"]["mean"] - stats[cls1]["r"]["std"],
            np.array(data[cls2]["r"]) <= stats[cls1]["r"]["mean"] + stats[cls1]["r"]["std"]
        )
        conf_matrix[i, j] = np.sum(within_1_sigma)

conf_matrix /= np.array([len(data[cls]["r"]) for cls in classes]).reshape(-1, 1)
print("Confusion Matrix:")
print(conf_matrix)

# Purity and Recall
purity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
print("Purity:", purity)
print("Recall:", recall)

# Pairplots
df = pd.read_csv(file_path)
sns.pairplot(df, hue='class')

# Filter DataFrame and Plot Pairplot
filtered_df = df[df['class'] != 'QSO'].drop('redshift', axis=1)
sns.pairplot(filtered_df, hue='class')

# Train/Test Split
X = filtered_df.drop('class', axis=1)
y = filtered_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# ROC Curve
y_test_bin = [0 if cls == "GALAXY" else 1 for cls in y_test]
fpr, tpr, _ = roc_curve(y_test_bin, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
