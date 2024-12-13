import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

file_path = 'skyserver-checkpoint.csv'

# create a dtype_dict of the csv so the parsing skips type inference increasing speed
dtype_dict = {
    "class": "category",  # 'category' reduces memory usage for repetitive strings
    "u": "float64",      
    "g": "float64",
    "r": "float64",
    "i": "float64",
    "z": "float64",
    "redshift": "float64"  # Scientific notation, so use float
}

df = pd.read_csv(file_path, dtype=dtype_dict)

# Separate data by class
star_data = df[df['class'] == 'STAR']
galaxy_data = df[df['class'] == 'GALAXY']
qso_data = df[df['class'] == 'QSO']

# Define features
features = ['u', 'g', 'r', 'i', 'z', 'redshift']

# Print header
print(f"{'':<10}{'u':<20}{'g':<20}{'r':<20}{'i':<20}{'z':<20}{'redshift':<20}")
print("-" * 130)

# Function to format mean±std for a given class DataFrame
def stats_line(cls_name, cls_data):
    line = f"{cls_name:<10}"
    for f in features:
        mean_val = cls_data[f].mean()
        std_val = cls_data[f].std()
        line += f"{mean_val:.5f}±{std_val:.5f} | "
    return line

print(stats_line("STAR", star_data))
print(stats_line("GALAXY", galaxy_data))
print(stats_line("QSO", qso_data))

# Example histogram plot for 'r' feature
sns.histplot(star_data['r'], color='blue', label='STAR', kde=True, alpha=0.6)
sns.histplot(galaxy_data['r'], color='green', label='GALAXY', kde=True, alpha=0.6)
sns.histplot(qso_data['r'], color='red', label='QSO', kde=True, alpha=0.6)
plt.xlabel('R-band Magnitude')
plt.ylabel('Frequency')
plt.title('Distribution of R-band Magnitude')
plt.legend(title='Object Type')
plt.show()
