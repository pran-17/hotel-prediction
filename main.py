import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import os
from scipy.stats import binomtest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# ------------------- Classification -------------------
file_path = "occupancy1.xlsx"
df = pd.read_excel(file_path)

# Drop any date/time columns
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        df.drop(columns=[col], inplace=True)

# Automatically detect target column
possible_targets = [col for col in df.columns if "occupancy" in col.lower()]
if not possible_targets:
    raise ValueError("❌ Could not find any target column related to occupancy.")
else:
    target_col = possible_targets[0]
    print(f"\n✅ Detected target column: '{target_col}'")

# Convert to binary classification: Occupied (1) vs Not Occupied (0)
y = (df[target_col] > 0).astype(int)
X = df.drop(columns=[target_col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------ ✅ Corrected Training and Prediction Block ------
# Train a lightweight Random Forest Classifier
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Predict probabilities safely
if len(clf.classes_) == 2:
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability for class 1
else:
    y_proba = np.zeros_like(y_pred, dtype=float)  # fallback
    print("⚠ Warning: Only one class detected during training. Probabilities set to 0.")
# --------------------------------------------------------

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Occupied", "Occupied"],
            yticklabels=["Not Occupied", "Occupied"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot 2: Feature Importance
importances = clf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Plot 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.tight_layout()
plt.show()

# ------------------- Clustering -------------------
file_path = "occupancy1.xlsx"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
elif not os.access(file_path, os.R_OK):
    print(f"File is not readable: {file_path}")
else:
    dataset = pd.read_excel(file_path)
    # Drop date/time columns
    for col in dataset.columns:
        if "date" in col.lower() or "time" in col.lower():
            dataset.drop(columns=[col], inplace=True, errors='ignore')
    # Select and scale numeric features
    features = dataset.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Manual WCSS Calculation
    print("\n📉 Manual WCSS Calculation for KMeans:")
    wcss_manual = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        centroids = kmeans.cluster_centers_
        wcss_k = 0
        for i in range(k):
            cluster_points = features_scaled[labels == i]
            distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
            wcss_k += np.sum(distances**2)
        wcss_manual.append(wcss_k)
        print(f"K = {k} -> WCSS = {wcss_k:.2f}")

    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss_manual, marker='o', linestyle='--')
    plt.title('Manual Elbow Method (WCSS)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # KMeans Clustering
    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_labels = kmeans_final.fit_predict(features_scaled)
    centroids_final = kmeans_final.cluster_centers_

    # KMeans outputs
    unique_kmeans_labels = np.unique(kmeans_labels)
    print(f"\n📊 Unique KMeans Cluster Labels: {unique_kmeans_labels}")
    if len(unique_kmeans_labels) == 1:
        print("⚠ Warning: All points assigned to a single cluster! Check your data for variability.")
    else:
        print("\n📍 KMeans Final Centroids (K=3):")
        for i, centroid in enumerate(centroids_final):
            print(f"Cluster {i}: {centroid}")
        print("\n📊 Sample KMeans Cluster Assignments:")
        print(kmeans_labels[:10])

        # Plot KMeans clustering
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=features_scaled[:, 0], y=features_scaled[:, 1],
                        hue=kmeans_labels, palette='viridis')
        plt.title('KMeans Clustering Result')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.show()

    # Agglomerative Clustering
    max_samples = 1000
    sampled_idx = np.random.choice(features_scaled.shape[0], min(max_samples, features_scaled.shape[0]), replace=False)
    sampled_data = features_scaled[sampled_idx]

    linkage_matrix = linkage(sampled_data, method='ward')
    print(f"\n🧩 Hierarchical Clustering Linkage Matrix Sample (Top 5 rows):")
    print(linkage_matrix[:5])

    # Dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Dendrogram - Hierarchical Clustering (Ward)')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agg_labels = agg.fit_predict(sampled_data)
    print("\n📊 Sample Agglomerative Cluster Assignments:")
    print(agg_labels[:10])

    # Plot Agglomerative Clustering
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=sampled_data[:, 0], y=sampled_data[:, 1],
                    hue=agg_labels, palette='coolwarm')
    plt.title('Agglomerative Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# ------------------- Time Series Forecasting -------------------
file_path = "occupancy1.xlsx"
df = pd.read_excel(file_path)

# Parse datetime
df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# Resample data hourly
df_hourly = df['Room_Occupancy_Count'].resample('H').mean().fillna(0)

# Split train & test
split_index = int(len(df_hourly) * 0.8)
train = df_hourly[:split_index]
test = df_hourly[split_index:]

# Fit ARIMA Model
model = ARIMA(train, order=(2, 1, 2))  # (p,d,q)
model_fit = model.fit()

# Model summary
print("\n=== ARIMA Model Summary ===")
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Evaluation
mse = mean_squared_error(test, forecast)
print(f"Test MSE: {mse:.2f}")

# Plot: Actual vs Forecast
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train", color='blue')
plt.plot(test.index, test, label="Actual", color='black')
plt.plot(forecast.index, forecast, label="Forecast", color='red')
plt.title("Room Occupancy Forecast (Hourly)")
plt.xlabel("Time")
plt.ylabel("Occupancy Count")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Plot: Forecast Residuals
residuals = test - forecast
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Forecast Residuals")
plt.xlabel("Time")
plt.ylabel("Error")
plt.grid()
plt.tight_layout()
plt.show()
#----sign test-------

df = pd.read_excel("C:\\Users\\Praneeth Ashokkumar\\OneDrive\\Desktop\\PA\\occupancy1.xlsx")

df.columns = [c.strip() for c in df.columns]


occupied_temp = df[df['Room_Occupancy_Count'] > 0]['S1_Temp']
not_occupied_temp = df[df['Room_Occupancy_Count'] == 0]['S1_Temp']


min_len = min(len(occupied_temp), len(not_occupied_temp))
occupied_sample = occupied_temp.sample(min_len, random_state=42)
not_occupied_sample = not_occupied_temp.sample(min_len, random_state=42)


diff = occupied_sample.values - not_occupied_sample.values


num_positive = (diff > 0).sum()
num_negative = (diff < 0).sum()


n = num_positive + num_negative
result =binomtest(num_positive, n=n, p=0.5, alternative='two-sided')


print("\n🧪 Sign Test (Temperature Occupied vs Not Occupied):")
print(f"Number of positive differences: {num_positive}")
print(f"Number of negative differences: {num_negative}")
print(f"p-value: {result.pvalue:.5f}")

if result.pvalue < 0.05:
    print("✅ Significant difference: Temperature tends to differ when occupied vs not occupied.")
else:
    print("❌ No significant temperature difference between occupied and not occupied rooms.")


plot_data = df[['Room_Occupancy_Count', 'S1_Temp']].copy()
plot_data.rename(columns={'Room_Occupancy_Count': 'Occupancy'}, inplace=True)


plt.figure(figsize=(8, 6))
sns.boxplot(data=plot_data, x='Occupancy', y='S1_Temp', palette="Set2")
plt.title('📦 Boxplot of Temperature vs Room Occupancy')
plt.xlabel('Occupancy (0 = Not Occupied, 1 = Occupied)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()