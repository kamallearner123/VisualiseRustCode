// Python Main application logic

document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupEventListeners();
    
    console.log('Python Programming Editor initialized');
});

function setupEventListeners() {
    // Run button
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runPythonCode);
    }
    
    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            clearEditor();
            clearOutput();
            clearPlots();
        });
    }
    
    // Example button
    const exampleBtn = document.getElementById('exampleBtn');
    if (exampleBtn) {
        exampleBtn.addEventListener('click', showPythonExampleMenu);
    }
    
    // Clear output button
    const clearOutputBtn = document.getElementById('clearOutputBtn');
    if (clearOutputBtn) {
        clearOutputBtn.addEventListener('click', clearOutput);
    }
    
    // Clear plots button
    const clearPlotsBtn = document.getElementById('clearPlotsBtn');
    if (clearPlotsBtn) {
        clearPlotsBtn.addEventListener('click', clearPlots);
    }
}

function showPythonExampleMenu() {
    const exampleCategories = [
        {
            name: 'Machine Learning Basics',
            examples: [
                { name: 'Linear Regression', key: 'ml_linear_regression' },
                { name: 'Logistic Regression', key: 'ml_logistic_regression' },
                { name: 'Decision Trees', key: 'ml_decision_trees' },
                { name: 'Random Forest', key: 'ml_random_forest' },
                { name: 'K-Means Clustering', key: 'ml_kmeans' },
                { name: 'SVM Classification', key: 'ml_svm' },
                { name: 'Naive Bayes', key: 'ml_naive_bayes' },
                { name: 'K-Nearest Neighbors', key: 'ml_knn' },
                { name: 'Gradient Boosting', key: 'ml_gradient_boosting' },
                { name: 'AdaBoost', key: 'ml_adaboost' },
                { name: 'Principal Component Analysis', key: 'ml_pca' },
                { name: 'DBSCAN Clustering', key: 'ml_dbscan' }
            ]
        },
        {
            name: 'Advanced Machine Learning',
            examples: [
                { name: 'Cross-Validation', key: 'ml_cross_validation' },
                { name: 'Grid Search Hyperparameters', key: 'ml_grid_search' },
                { name: 'Feature Selection', key: 'ml_feature_selection' },
                { name: 'Ensemble Methods', key: 'ml_ensemble' },
                { name: 'Pipeline Creation', key: 'ml_pipeline' },
                { name: 'Model Evaluation Metrics', key: 'ml_metrics' },
                { name: 'Imbalanced Data Handling', key: 'ml_imbalanced' },
                { name: 'Anomaly Detection', key: 'ml_anomaly' }
            ]
        },
        {
            name: 'Regression Models',
            examples: [
                { name: 'Ridge Regression', key: 'ml_ridge' },
                { name: 'Lasso Regression', key: 'ml_lasso' },
                { name: 'Polynomial Regression', key: 'ml_polynomial' },
                { name: 'ElasticNet Regression', key: 'ml_elasticnet' }
            ]
        },
        {
            name: 'Deep Learning',
            examples: [
                { name: 'Neural Network (Basic)', key: 'dl_neural_network' },
                { name: 'CNN - Image Classification', key: 'dl_cnn' },
                { name: 'RNN - Time Series', key: 'dl_rnn' },
                { name: 'Transfer Learning', key: 'dl_transfer_learning' }
            ]
        },
        {
            name: 'Data Science',
            examples: [
                { name: 'Pandas DataFrame Basics', key: 'ds_pandas_basics' },
                { name: 'Data Cleaning', key: 'ds_data_cleaning' },
                { name: 'Data Visualization', key: 'ds_visualization' },
                { name: 'Statistical Analysis', key: 'ds_statistics' },
                { name: 'Feature Engineering', key: 'ds_feature_engineering' }
            ]
        },
        {
            name: 'NumPy & Arrays',
            examples: [
                { name: 'Array Operations', key: 'np_array_ops' },
                { name: 'Matrix Operations', key: 'np_matrix_ops' },
                { name: 'Broadcasting', key: 'np_broadcasting' },
                { name: 'Linear Algebra', key: 'np_linear_algebra' }
            ]
        },
        {
            name: 'Python Basics',
            examples: [
                { name: 'Variables & Types', key: 'py_basics' },
                { name: 'Lists & Dictionaries', key: 'py_collections' },
                { name: 'Functions', key: 'py_functions' },
                { name: 'Classes & OOP', key: 'py_classes' },
                { name: 'File I/O', key: 'py_file_io' }
            ]
        }
    ];
    
    const menu = document.createElement('div');
    menu.className = 'example-menu';
    
    const header = document.createElement('h4');
    header.innerHTML = '<i class="fas fa-brain"></i> Python & Machine Learning Examples';
    menu.appendChild(header);
    
    const categoriesContainer = document.createElement('div');
    categoriesContainer.className = 'example-categories';
    
    exampleCategories.forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'example-category';
        
        const categoryHeader = document.createElement('div');
        categoryHeader.className = 'category-header';
        categoryHeader.innerHTML = `${category.name} <i class="fas fa-chevron-down"></i>`;
        categoryHeader.onclick = function() {
            this.classList.toggle('collapsed');
            exampleGrid.style.display = this.classList.contains('collapsed') ? 'none' : 'grid';
        };
        
        const exampleGrid = document.createElement('div');
        exampleGrid.className = 'example-grid';
        
        category.examples.forEach(ex => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-secondary btn-sm example-item';
            btn.textContent = ex.name;
            btn.onclick = () => {
                loadPythonExample(ex.key);
                menu.remove();
            };
            exampleGrid.appendChild(btn);
        });
        
        categoryDiv.appendChild(categoryHeader);
        categoryDiv.appendChild(exampleGrid);
        categoriesContainer.appendChild(categoryDiv);
    });
    
    menu.appendChild(categoriesContainer);
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn btn-secondary';
    closeBtn.textContent = 'Cancel';
    closeBtn.onclick = () => menu.remove();
    menu.appendChild(closeBtn);
    
    document.body.appendChild(menu);
}

function loadPythonExample(key) {
    const examples = {
        'ml_linear_regression': `# Linear Regression Example with Visualization
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")
print(f"Coefficients: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Linear Regression (R² = {score:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_logistic_regression': `# Logistic Regression Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Use only 2 classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))
`,
        'ml_decision_trees': `# Decision Tree Classifier Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Tree Depth: {dt.get_depth()}")
print(f"Number of Leaves: {dt.get_n_leaves()}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
`,
        'ml_random_forest': `# Random Forest Classifier Example
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of Trees: {rf.n_estimators}")

# Feature importance
print("\\nTop 5 Important Features:")
feature_importance = sorted(zip(wine.feature_names, rf.feature_importances_), 
                           key=lambda x: x[1], reverse=True)[:5]
for name, importance in feature_importance:
    print(f"{name}: {importance:.4f}")
`,
        'ml_kmeans': `# K-Means Clustering Example with Visualization
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Evaluate
silhouette_avg = silhouette_score(X, y_pred)
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"\\nCluster Centers:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: [{center[0]:.2f}, {center[1]:.2f}]")

# Count samples per cluster
unique, counts = np.unique(y_pred, return_counts=True)
print(f"\\nSamples per cluster: {dict(zip(unique, counts))}")

# Visualize clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=300, edgecolors='black', linewidths=2, 
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-Means Clustering (Silhouette Score = {silhouette_avg:.4f})')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_svm': `# Support Vector Machine (SVM) Example
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Support Vectors: {len(svm.support_vectors_)}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
`,
        'dl_neural_network': `# Simple Neural Network with TensorFlow/Keras
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Note: TensorFlow may not be installed
# This is a demonstration of the code structure

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dataset prepared:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Uncomment to use with TensorFlow:
# import tensorflow as tf
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
`,
        'ds_pandas_basics': `# Pandas DataFrame Basics
import pandas as pd
import numpy as np

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 55000, 65000, 58000]
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)
print("\\nDataFrame Info:")
print(df.info())
print("\\nBasic Statistics:")
print(df.describe())
print("\\nColumn Names:")
print(df.columns.tolist())
print("\\nFirst 3 rows:")
print(df.head(3))
print("\\nFiltering (Age > 28):")
print(df[df['Age'] > 28])
`,
        'ds_data_cleaning': `# Data Cleaning Example
import pandas as pd
import numpy as np

# Create dataset with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': ['a', 'b', 'c', 'd', 'e']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print(f"\\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values
df_filled = df.copy()
df_filled['A'].fillna(df_filled['A'].mean(), inplace=True)
df_filled['B'].fillna(df_filled['B'].median(), inplace=True)
df_filled['C'].fillna(0, inplace=True)

print("\\nCleaned DataFrame:")
print(df_filled)

# Remove duplicates
df_no_duplicates = df_filled.drop_duplicates()
print(f"\\nShape after removing duplicates: {df_no_duplicates.shape}")
`,
        'ds_visualization': `# Data Visualization Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(100, 500, 100),
    'Revenue': np.random.randint(1000, 5000, 100)
})

print("Sales Data:")
print(data.head(10))
print("\\nSummary Statistics:")
print(data[['Sales', 'Revenue']].describe())

# Calculate correlations
correlation = data[['Sales', 'Revenue']].corr()
print("\\nCorrelation Matrix:")
print(correlation)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
axes[0, 0].plot(data['Date'], data['Sales'], label='Sales', color='blue')
axes[0, 0].set_title('Sales Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Bar plot for monthly average
data['Month'] = data['Date'].dt.to_period('M')
monthly = data.groupby('Month')[['Sales', 'Revenue']].mean()
monthly.plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'lightcoral'])
axes[0, 1].set_title('Average Monthly Sales & Revenue')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()

# Scatter plot
axes[1, 0].scatter(data['Sales'], data['Revenue'], alpha=0.5, color='green')
axes[1, 0].set_title('Sales vs Revenue')
axes[1, 0].set_xlabel('Sales')
axes[1, 0].set_ylabel('Revenue')
axes[1, 0].grid(True, alpha=0.3)

# Histogram
axes[1, 1].hist(data['Sales'], bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Sales Distribution')
axes[1, 1].set_xlabel('Sales')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nVisualization created successfully!")
`,
        'ds_statistics': `# Statistical Analysis Example
import pandas as pd
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

# Descriptive statistics
print("Group A Statistics:")
print(f"Mean: {np.mean(group_a):.2f}")
print(f"Median: {np.median(group_a):.2f}")
print(f"Std Dev: {np.std(group_a):.2f}")
print(f"Variance: {np.var(group_a):.2f}")

print("\\nGroup B Statistics:")
print(f"Mean: {np.mean(group_b):.2f}")
print(f"Median: {np.median(group_b):.2f}")
print(f"Std Dev: {np.std(group_b):.2f}")

# T-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\\nT-test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
`,
        'ds_feature_engineering': `# Feature Engineering Example
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create sample dataset
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 75000, 80000, 90000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'years_exp': [2, 5, 10, 15, 20]
})

print("Original Data:")
print(data)

# Create new features
data['income_per_year'] = data['income'] / data['years_exp']
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])

print("\\nWith New Features:")
print(data)

# Encode categorical variables
le = LabelEncoder()
data['education_encoded'] = le.fit_transform(data['education'])

# Scale numerical features
scaler = StandardScaler()
data[['age_scaled', 'income_scaled']] = scaler.fit_transform(data[['age', 'income']])

print("\\nWith Encoding and Scaling:")
print(data)
`,
        'np_array_ops': `# NumPy Array Operations
import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print("Array 1:", arr1)
print("Array 2:", arr2)

# Basic operations
print("\\nAddition:", arr1 + arr2)
print("Multiplication:", arr1 * arr2)
print("Power:", arr1 ** 2)

# Statistical operations
print("\\nMean:", arr1.mean())
print("Sum:", arr1.sum())
print("Min:", arr1.min())
print("Max:", arr1.max())
print("Std Dev:", arr1.std())

# Array manipulation
print("\\nReshaped (5,1):")
print(arr1.reshape(5, 1))

# 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\\n2D Array:")
print(arr_2d)
print("Shape:", arr_2d.shape)
print("Sum of columns:", arr_2d.sum(axis=0))
print("Sum of rows:", arr_2d.sum(axis=1))
`,
        'np_matrix_ops': `# NumPy Matrix Operations
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\\nMatrix B:")
print(B)

# Matrix operations
print("\\nMatrix Multiplication (A @ B):")
print(A @ B)

print("\\nElement-wise Multiplication (A * B):")
print(A * B)

print("\\nTranspose of A:")
print(A.T)

print("\\nDeterminant of A:")
print(np.linalg.det(A))

print("\\nInverse of A:")
print(np.linalg.inv(A))

print("\\nEigenvalues and Eigenvectors of A:")
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
`,
        'np_broadcasting': `# NumPy Broadcasting Example
import numpy as np

# Broadcasting with 1D arrays
a = np.array([1, 2, 3])
b = 10
print("Array:", a)
print("Scalar:", b)
print("Broadcasting (a + b):", a + b)

# Broadcasting with 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([10, 20, 30])

print("\\nMatrix:")
print(matrix)
print("\\nRow vector:", row)
print("\\nBroadcasting (matrix + row):")
print(matrix + row)

# Column broadcasting
col = np.array([[100], [200], [300]])
print("\\nColumn vector:")
print(col)
print("\\nBroadcasting (matrix + col):")
print(matrix + col)

# Complex broadcasting
result = matrix * row + col
print("\\nComplex broadcasting (matrix * row + col):")
print(result)
`,
        'np_linear_algebra': `# Linear Algebra with NumPy
import numpy as np

# System of linear equations: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

print("Matrix A:")
print(A)
print("\\nVector b:", b)

# Solve linear system
x = np.linalg.solve(A, b)
print("\\nSolution x:", x)
print("Verification (A @ x):", A @ x)

# Matrix properties
print("\\nMatrix Properties:")
print("Determinant:", np.linalg.det(A))
print("Rank:", np.linalg.matrix_rank(A))
print("Condition number:", np.linalg.cond(A))

# Norms
print("\\nNorms:")
print("L1 norm of b:", np.linalg.norm(b, 1))
print("L2 norm of b:", np.linalg.norm(b, 2))
print("Infinity norm of b:", np.linalg.norm(b, np.inf))

# SVD
U, s, Vt = np.linalg.svd(A)
print("\\nSingular Values:", s)
`,
        'py_basics': `# Python Basics - Variables & Types
# This is a comment

# Variables and basic types
name = "Python"
version = 3.11
is_awesome = True
pi = 3.14159

print(f"Language: {name}")
print(f"Version: {version}")
print(f"Is awesome? {is_awesome}")
print(f"Pi value: {pi}")

# Type checking
print(f"\\nType of name: {type(name)}")
print(f"Type of version: {type(version)}")
print(f"Type of is_awesome: {type(is_awesome)}")

# Basic operations
x = 10
y = 3
print(f"\\nMath Operations:")
print(f"{x} + {y} = {x + y}")
print(f"{x} - {y} = {x - y}")
print(f"{x} * {y} = {x * y}")
print(f"{x} / {y} = {x / y:.2f}")
print(f"{x} // {y} = {x // y}")
print(f"{x} % {y} = {x % y}")
print(f"{x} ** {y} = {x ** y}")

# String operations
greeting = "Hello"
target = "World"
message = greeting + " " + target + "!"
print(f"\\n{message}")
print(f"Length: {len(message)}")
print(f"Uppercase: {message.upper()}")
print(f"Lowercase: {message.lower()}")
`,
        'py_collections': `# Python Collections - Lists & Dictionaries

# Lists
fruits = ["apple", "banana", "orange", "grape"]
print("Fruits:", fruits)
print("First fruit:", fruits[0])
print("Last fruit:", fruits[-1])

# List operations
fruits.append("mango")
print("After append:", fruits)
fruits.remove("banana")
print("After remove:", fruits)
print("Length:", len(fruits))

# List comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print("\\nNumbers:", numbers)
print("Squares:", squares)

# Dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "Machine Learning", "Data Science"]
}
print("\\nPerson:", person)
print("Name:", person["name"])
print("Skills:", person["skills"])

# Dictionary operations
person["email"] = "alice@example.com"
print("\\nAfter adding email:", person)

# Iterating
print("\\nIterating over dictionary:")
for key, value in person.items():
    print(f"{key}: {value}")
`,
        'py_functions': `# Python Functions

# Simple function
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
print(greet("Bob"))

# Function with default parameters
def power(base, exponent=2):
    return base ** exponent

print(f"\\n2^3 = {power(2, 3)}")
print(f"5^2 = {power(5)}")

# Function with multiple return values
def calculate_stats(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)
    return total, average, maximum, minimum

data = [10, 20, 30, 40, 50]
total, avg, max_val, min_val = calculate_stats(data)
print(f"\\nData: {data}")
print(f"Total: {total}")
print(f"Average: {avg}")
print(f"Max: {max_val}")
print(f"Min: {min_val}")

# Lambda functions
square = lambda x: x ** 2
add = lambda x, y: x + y

print(f"\\nSquare of 5: {square(5)}")
print(f"Add 3 and 7: {add(3, 7)}")

# Map and filter
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(f"\\nOriginal: {numbers}")
print(f"Doubled: {doubled}")
print(f"Even numbers: {evens}")
`,
        'py_classes': `# Python Classes and OOP

# Define a class
class Dog:
    # Class variable
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def get_info(self):
        return f"{self.name} is {self.age} years old"

# Create objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())
print(dog2.bark())
print(dog1.get_info())
print(dog2.get_info())
print(f"Species: {Dog.species}")

# Inheritance
class GoldenRetriever(Dog):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color
    
    def fetch(self):
        return f"{self.name} is fetching the ball!"
    
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info} and has {self.color} fur"

golden = GoldenRetriever("Charlie", 4, "golden")
print(f"\\n{golden.get_info()}")
print(golden.bark())
print(golden.fetch())
`,
        'py_file_io': `# Python File I/O Example

# Writing to a file
content = """This is a sample text file.
It contains multiple lines.
Python makes file handling easy!"""

try:
    # Write to file
    with open('sample.txt', 'w') as file:
        file.write(content)
    print("File written successfully!")
    
    # Read from file
    with open('sample.txt', 'r') as file:
        data = file.read()
    print("\\nFile contents:")
    print(data)
    
    # Read line by line
    print("\\nReading line by line:")
    with open('sample.txt', 'r') as file:
        for i, line in enumerate(file, 1):
            print(f"Line {i}: {line.strip()}")
    
    # Append to file
    with open('sample.txt', 'a') as file:
        file.write("\\nThis line was appended!")
    
    print("\\nAfter appending:")
    with open('sample.txt', 'r') as file:
        print(file.read())
    
except IOError as e:
    print(f"An error occurred: {e}")

print("\\nNote: File operations work with proper permissions.")
`,
        'ml_naive_bayes': `# Naive Bayes Classification
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict
y_pred = nb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\\nClass Priors: {nb.class_prior_}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\\nConfusion Matrix:\\n{cm}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title(f'Naive Bayes Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
`,
        'ml_knn': `# K-Nearest Neighbors (KNN) Classification
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Use only first 4 classes for simplicity
mask = y < 4
X, y = X[mask], y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different K values
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# Best K
best_k = k_values[np.argmax(test_scores)]
print(f"Best K: {best_k}")
print(f"Best Test Accuracy: {max(test_scores):.4f}")

# Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot K vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(k_values, test_scores, 's-', label='Test Score', linewidth=2)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN: Model Performance vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_gradient_boosting': `# Gradient Boosting Classifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)
y_pred_proba = gb.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\\nFeature Importance (Top 5):")
feature_importance = sorted(zip(range(20), gb.feature_importances_), 
                           key=lambda x: x[1], reverse=True)[:5]
for idx, importance in feature_importance:
    print(f"Feature {idx}: {importance:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Receiver Operating Characteristic (ROC)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature Importance
importances = gb.feature_importances_
indices = np.argsort(importances)[-10:]
axes[1].barh(range(len(indices)), importances[indices], color='skyblue')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([f'Feature {i}' for i in indices])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importances')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
`,
        'ml_adaboost': `# AdaBoost Classifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost with different number of estimators
n_estimators_list = [10, 25, 50, 100, 200]
train_scores = []
test_scores = []

for n in n_estimators_list:
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))

# Best model
best_n = n_estimators_list[np.argmax(test_scores)]
ada_best = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=best_n,
    random_state=42
)
ada_best.fit(X_train, y_train)
y_pred = ada_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best n_estimators: {best_n}")
print(f"Best Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\\nConfusion Matrix:\\n{cm}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance vs n_estimators
axes[0].plot(n_estimators_list, train_scores, 'o-', label='Training', linewidth=2)
axes[0].plot(n_estimators_list, test_scores, 's-', label='Testing', linewidth=2)
axes[0].axvline(best_n, color='red', linestyle='--', label=f'Best n={best_n}')
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('AdaBoost Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
axes[1].set_title(f'Confusion Matrix (Acc: {accuracy:.4f})')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()
`,
        'ml_pca': `# Principal Component Analysis (PCA)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

print("Explained Variance Ratio:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f} ({cumsum_var[i]:.4f} cumulative)")

# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(explained_var) + 1), explained_var, 
           alpha=0.7, label='Individual', color='skyblue')
axes[0].plot(range(1, len(explained_var) + 1), cumsum_var, 
            'ro-', linewidth=2, label='Cumulative')
axes[0].axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D projection
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    mask = y == i
    axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   c=color, label=iris.target_names[i], 
                   alpha=0.7, edgecolors='k')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
axes[1].set_title('2D PCA Projection')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_dbscan': `# DBSCAN Clustering
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data (non-linear clusters)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"\\nCluster sizes:")
unique, counts = np.unique(labels[labels != -1], return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} points")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
               alpha=0.6, edgecolors='k')
axes[0].set_title('Original Data (True Labels)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# DBSCAN results
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k')
# Highlight noise points
noise_mask = labels == -1
if n_noise > 0:
    axes[1].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                   c='red', marker='x', s=100, label='Noise')
axes[1].set_title(f'DBSCAN Clustering ({n_clusters} clusters, {n_noise} noise)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label='Cluster')
plt.tight_layout()
plt.show()
`,
        'ml_cross_validation': `# Cross-Validation Example
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Different CV strategies
cv_strategies = {
    'KFold (5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'KFold (10)': KFold(n_splits=10, shuffle=True, random_state=42),
    'StratifiedKFold (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
}

results = {}
for name, cv in cv_strategies.items():
    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"{name}:")
    print(f"  Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print()

# Detailed cross-validation
cv_results = cross_validate(rf, X, y, cv=5, 
                           scoring=['accuracy', 'precision_macro', 'recall_macro'],
                           return_train_score=True)

print("Detailed 5-Fold CV Results:")
print(f"Train Accuracy: {cv_results['train_accuracy'].mean():.4f}")
print(f"Test Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Test Precision: {cv_results['test_precision_macro'].mean():.4f}")
print(f"Test Recall: {cv_results['test_recall_macro'].mean():.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of CV scores
axes[0].boxplot([results[name] for name in results.keys()], 
               labels=list(results.keys()))
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Cross-Validation Score Distribution')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=15)

# Train vs Test scores
fold_nums = range(1, 6)
axes[1].plot(fold_nums, cv_results['train_accuracy'], 'o-', 
            label='Training', linewidth=2, markersize=8)
axes[1].plot(fold_nums, cv_results['test_accuracy'], 's-', 
            label='Testing', linewidth=2, markersize=8)
axes[1].axhline(cv_results['test_accuracy'].mean(), 
               color='red', linestyle='--', label='Mean Test')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train vs Test Accuracy per Fold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_grid_search': `# Grid Search for Hyperparameter Tuning
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

# Grid Search
print("Performing Grid Search...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Results
print(f"\\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# Top 5 combinations
results = grid_search.cv_results_
top_indices = np.argsort(results['mean_test_score'])[-5:][::-1]
print("\\nTop 5 Parameter Combinations:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. Score: {results['mean_test_score'][idx]:.4f}, "
          f"Params: {results['params'][idx]}")

# Visualize for RBF kernel
rbf_results = [(res['params'], res['mean_test_score']) 
               for res in [dict(zip(results.keys(), values)) 
                          for values in zip(*results.values())]
               if res['params']['kernel'] == 'rbf']

C_values = sorted(set(r[0]['C'] for r in rbf_results))
gamma_values = sorted(set(r[0]['gamma'] for r in rbf_results))

# Create heatmap data
heatmap_data = np.zeros((len(gamma_values), len(C_values)))
for params, score in rbf_results:
    i = gamma_values.index(params['gamma'])
    j = C_values.index(params['C'])
    heatmap_data[i, j] = score

# Plot heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Mean CV Accuracy')
plt.xticks(range(len(C_values)), C_values)
plt.yticks(range(len(gamma_values)), gamma_values)
plt.xlabel('C (Regularization)')
plt.ylabel('Gamma')
plt.title('Grid Search Results (RBF Kernel)\\nCV Accuracy Heatmap')

# Annotate cells with values
for i in range(len(gamma_values)):
    for j in range(len(C_values)):
        text = plt.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="white", fontsize=9)

plt.tight_layout()
plt.show()
`,
        'ml_feature_selection': `# Feature Selection Techniques
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original features: {X.shape[1]}")

# Method 1: SelectKBest (Univariate)
selector_kbest = SelectKBest(f_classif, k=10)
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
X_test_kbest = selector_kbest.transform(X_test)

# Method 2: RFE (Recursive Feature Elimination)
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Method 3: Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
top_10_idx = np.argsort(importances)[-10:]
X_train_rf = X_train[:, top_10_idx]
X_test_rf = X_test[:, top_10_idx]

# Compare performance
methods = {
    'All Features': (X_train, X_test),
    'SelectKBest': (X_train_kbest, X_test_kbest),
    'RFE': (X_train_rfe, X_test_rfe),
    'RF Importance': (X_train_rf, X_test_rf)
}

results = {}
for name, (X_tr, X_te) in methods.items():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_train)
    score = clf.score(X_te, y_test)
    results[name] = score
    print(f"{name}: {score:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Method comparison
axes[0].bar(results.keys(), results.values(), color=['gray', 'skyblue', 'lightgreen', 'coral'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Feature Selection Methods Comparison')
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(results['All Features'], color='red', linestyle='--', label='Baseline')
axes[0].legend()

# Feature importance
top_indices = np.argsort(importances)[-15:]
axes[1].barh(range(len(top_indices)), importances[top_indices], color='steelblue')
axes[1].set_yticks(range(len(top_indices)))
axes[1].set_yticklabels([cancer.feature_names[i] for i in top_indices], fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 15 Feature Importances (Random Forest)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
`,
        'ml_ensemble': `# Ensemble Methods - Voting Classifier
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Load data
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)
clf4 = DecisionTreeClassifier(max_depth=5, random_state=42)

# Hard voting ensemble
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dt', clf4)],
    voting='hard'
)

# Soft voting ensemble
voting_soft = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dt', clf4)],
    voting='soft'
)

# Train and evaluate all models
models = {
    'Logistic Regression': clf1,
    'Random Forest': clf2,
    'SVM': clf3,
    'Decision Tree': clf4,
    'Hard Voting': voting_hard,
    'Soft Voting': voting_soft
}

results = {}
cv_scores = {}

print("Model Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    results[name] = {'train': train_score, 'test': test_score, 'cv': cv_score}
    print(f"{name}:")
    print(f"  Train: {train_score:.4f}, Test: {test_score:.4f}, CV: {cv_score:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance comparison
model_names = list(results.keys())
test_scores = [results[name]['test'] for name in model_names]
cv_scores = [results[name]['cv'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, test_scores, width, label='Test Score', color='skyblue')
axes[0].bar(x + width/2, cv_scores, width, label='CV Score', color='lightcoral')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Improvement over individual classifiers
individual_scores = [results[name]['test'] for name in list(results.keys())[:4]]
ensemble_scores = [results[name]['test'] for name in ['Hard Voting', 'Soft Voting']]
mean_individual = np.mean(individual_scores)

axes[1].bar(['Mean Individual', 'Hard Voting', 'Soft Voting'], 
           [mean_individual] + ensemble_scores,
           color=['gray', 'lightgreen', 'lightblue'])
axes[1].axhline(mean_individual, color='red', linestyle='--', linewidth=2, 
               label='Mean Individual')
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Ensemble vs Individual Classifiers')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
`,
        'ml_pipeline': `# ML Pipeline Creation
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for pipeline
param_grid = {
    'feature_selection__k': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None]
}

# Grid search on pipeline
print("Performing Grid Search on Pipeline...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Results
print(f"\\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# Get feature selection results
best_pipeline = grid_search.best_estimator_
selected_features = best_pipeline.named_steps['feature_selection'].get_support()
print(f"\\nNumber of selected features: {selected_features.sum()}")
print(f"Selected feature names:")
for i, selected in enumerate(selected_features):
    if selected:
        print(f"  - {cancer.feature_names[i]}")

# Compare with different k values
k_values = [5, 10, 15, 20, 30]
scores = []

for k in k_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=min(k, X.shape[1]))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    scores.append(pipe.score(X_test, y_test))

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, 'o-', linewidth=2, markersize=10, color='steelblue')
best_k = grid_search.best_params_['feature_selection__k']
best_score = grid_search.score(X_test, y_test)
plt.axvline(best_k, color='red', linestyle='--', linewidth=2, 
           label=f'Best k={best_k}')
plt.axhline(best_score, color='green', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Number of Features (k)')
plt.ylabel('Test Accuracy')
plt.title('Pipeline Performance vs Number of Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nPipeline steps: {[name for name, _ in pipeline.steps]}")
`,
        'ml_ridge': `# Ridge Regression (L2 Regularization)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate data with noise
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []
coefficients = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    coefficients.append(ridge.coef_[0])

# Best alpha
best_alpha = alphas[np.argmax(test_scores)]
print(f"Best alpha: {best_alpha}")
print(f"Best R² score: {max(test_scores):.4f}")

# Compare with Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print(f"\\nLinear Regression R²: {lr_score:.4f}")
print(f"Ridge Regression R²: {max(test_scores):.4f}")

# Best Ridge model
best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_train, y_train)
y_pred_ridge = best_ridge.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Alpha vs Score
axes[0].semilogx(alphas, train_scores, 'o-', label='Training', linewidth=2)
axes[0].semilogx(alphas, test_scores, 's-', label='Testing', linewidth=2)
axes[0].axvline(best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha}')
axes[0].set_xlabel('Alpha (Regularization Strength)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Ridge Regression: Alpha Tuning')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predictions comparison
axes[1].scatter(X_test, y_test, color='black', alpha=0.5, label='True values')
sorted_idx = X_test.flatten().argsort()
axes[1].plot(X_test[sorted_idx], y_pred_lr[sorted_idx], 
            color='blue', linewidth=2, label='Linear Regression')
axes[1].plot(X_test[sorted_idx], y_pred_ridge[sorted_idx], 
            color='red', linewidth=2, label=f'Ridge (α={best_alpha})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Predictions Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_lasso': `# Lasso Regression (L1 Regularization)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate data with multiple features
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, 
                      noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different alpha values
alphas = np.logspace(-4, 2, 50)
coefs = []
train_scores = []
test_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    train_scores.append(lasso.score(X_train, y_train))
    test_scores.append(lasso.score(X_test, y_test))

# Best alpha
best_alpha = alphas[np.argmax(test_scores)]
best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

print(f"Best alpha: {best_alpha:.4f}")
print(f"Best R² score: {max(test_scores):.4f}")
print(f"\\nFeature Coefficients:")
for i, coef in enumerate(best_lasso.coef_):
    print(f"Feature {i}: {coef:.4f}" + (" (eliminated)" if abs(coef) < 0.001 else ""))

# Compare with Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"\\nLinear Regression R²: {lr.score(X_test, y_test):.4f}")
print(f"Lasso R²: {max(test_scores):.4f}")
print(f"Number of non-zero coefficients: {np.sum(np.abs(best_lasso.coef_) > 0.001)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient paths
coefs = np.array(coefs)
for i in range(coefs.shape[1]):
    axes[0].semilogx(alphas, coefs[:, i], label=f'Feature {i}')
axes[0].axvline(best_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'Best α={best_alpha:.4f}')
axes[0].set_xlabel('Alpha (Regularization Strength)')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_title('Lasso Coefficient Paths')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Alpha vs Score
axes[1].semilogx(alphas, train_scores, label='Training', linewidth=2)
axes[1].semilogx(alphas, test_scores, label='Testing', linewidth=2)
axes[1].axvline(best_alpha, color='red', linestyle='--', 
               label=f'Best α={best_alpha:.4f}')
axes[1].set_xlabel('Alpha (Regularization Strength)')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Lasso Performance vs Alpha')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`,
        'ml_polynomial': `# Polynomial Regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 0.5 * X**2 - 3 * X + np.random.randn(100, 1) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different polynomial degrees
degrees = [1, 2, 3, 4, 5, 8, 10]
train_scores = []
test_scores = []
models = []

for degree in degrees:
    # Create pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    models.append(model)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Degree {degree}: Train R²={train_score:.4f}, Test R²={test_score:.4f}")

# Best degree
best_degree = degrees[np.argmax(test_scores)]
best_model = models[np.argmax(test_scores)]
print(f"\\nBest degree: {best_degree}")
print(f"Best test R² score: {max(test_scores):.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Polynomial fits
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
for i, degree in enumerate([1, 2, 3, 5]):
    ax = axes[i // 2, i % 2]
    model = models[degrees.index(degree)]
    y_plot = model.predict(X_plot)
    
    ax.scatter(X_train, y_train, alpha=0.5, label='Training data', s=30)
    ax.scatter(X_test, y_test, alpha=0.5, label='Test data', s=30, color='red')
    ax.plot(X_plot, y_plot, linewidth=2, label=f'Degree {degree}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Polynomial Degree {degree} (R²={test_scores[degrees.index(degree)]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
plt.plot(degrees, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
plt.axvline(best_degree, color='red', linestyle='--', label=f'Best degree={best_degree}')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Performance vs Polynomial Degree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
        'ml_elasticnet': `# ElasticNet Regression (L1 + L2)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate data
X, y = make_regression(n_samples=100, n_features=20, n_informative=10,
                      noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different l1_ratio values (0=Ridge, 1=Lasso)
l1_ratios = np.linspace(0, 1, 11)
alpha = 1.0

results = {'train': [], 'test': [], 'n_nonzero': []}

for l1_ratio in l1_ratios:
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    en.fit(X_train_scaled, y_train)
    
    results['train'].append(en.score(X_train_scaled, y_train))
    results['test'].append(en.score(X_test_scaled, y_test))
    results['n_nonzero'].append(np.sum(np.abs(en.coef_) > 0.001))

# Best l1_ratio
best_l1_ratio = l1_ratios[np.argmax(results['test'])]
print(f"Best l1_ratio: {best_l1_ratio:.2f}")
print(f"Best test R² score: {max(results['test']):.4f}")

# Compare with pure Lasso and Ridge
lasso = Lasso(alpha=alpha, max_iter=10000)
ridge = Ridge(alpha=alpha)

lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

print(f"\\nComparison (alpha={alpha}):")
print(f"Ridge R²: {ridge.score(X_test_scaled, y_test):.4f}")
print(f"ElasticNet R²: {max(results['test']):.4f}")
print(f"Lasso R²: {lasso.score(X_test_scaled, y_test):.4f}")

# Train best model
best_en = ElasticNet(alpha=alpha, l1_ratio=best_l1_ratio, max_iter=10000)
best_en.fit(X_train_scaled, y_train)

print(f"\\nNon-zero coefficients: {np.sum(np.abs(best_en.coef_) > 0.001)}/{len(best_en.coef_)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance vs l1_ratio
axes[0].plot(l1_ratios, results['train'], 'o-', label='Training', linewidth=2)
axes[0].plot(l1_ratios, results['test'], 's-', label='Testing', linewidth=2)
axes[0].axvline(best_l1_ratio, color='red', linestyle='--', 
               label=f'Best l1_ratio={best_l1_ratio:.2f}')
axes[0].axvline(0, color='blue', linestyle=':', alpha=0.5, label='Ridge')
axes[0].axvline(1, color='green', linestyle=':', alpha=0.5, label='Lasso')
axes[0].set_xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
axes[0].set_ylabel('R² Score')
axes[0].set_title(f'ElasticNet Performance (alpha={alpha})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature sparsity vs l1_ratio
ax2 = axes[0].twinx()
ax2.plot(l1_ratios, results['n_nonzero'], 'o--', color='orange', 
        label='Non-zero features', alpha=0.7)
ax2.set_ylabel('Number of Non-zero Features', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Coefficient comparison
coef_ridge = ridge.coef_
coef_en = best_en.coef_
coef_lasso = lasso.coef_

x = np.arange(len(coef_ridge))
width = 0.25

axes[1].bar(x - width, coef_ridge, width, label='Ridge', alpha=0.8)
axes[1].bar(x, coef_en, width, label=f'ElasticNet (l1={best_l1_ratio:.2f})', alpha=0.8)
axes[1].bar(x + width, coef_lasso, width, label='Lasso', alpha=0.8)
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('Coefficient Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
`
    };
    
    const code = examples[key] || '# Example not found';
    setEditorCode(code);
}

async function runPythonCode() {
    const code = getEditorCode();
    
    if (!code.trim()) {
        showOutput('Error: No code to execute', 'error');
        return;
    }
    
    // Show loading
    showLoading(true);
    clearOutput();
    clearPlots();
    
    try {
        const response = await fetch('/python/execute/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code: code })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            handlePythonExecutionResult(result);
        } else {
            showOutput(`Error: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showOutput(`Network Error: ${error.message}`, 'error');
        console.error('Execution error:', error);
    } finally {
        showLoading(false);
    }
}

function handlePythonExecutionResult(result) {
    let output = '';
    
    if (result.stdout) {
        output += `<div class="output-section output-stdout">
            <div class="output-label">Output:</div>
            <pre>${escapeHtml(result.stdout)}</pre>
        </div>`;
    }
    
    if (result.stderr) {
        output += `<div class="output-section output-stderr">
            <div class="output-label">Errors:</div>
            <pre>${escapeHtml(result.stderr)}</pre>
        </div>`;
    }
    
    if (!result.stdout && !result.stderr) {
        output += '<div class="output-section"><em>No output</em></div>';
    }
    
    showOutput(output, result.success ? 'success' : 'error');
    
    // Display plots if any
    if (result.plots && result.plots.length > 0) {
        displayPlots(result.plots);
    } else {
        showPlotsPlaceholder();
    }
}

function displayPlots(plots) {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    
    let plotsHtml = '';
    plots.forEach((plot, index) => {
        plotsHtml += `
            <div class="plot-container">
                <div class="plot-title">Plot ${index + 1}</div>
                <img src="data:image/png;base64,${plot.data}" alt="Plot ${index + 1}">
            </div>
        `;
    });
    
    plotsDiv.innerHTML = plotsHtml;
}

function showPlotsPlaceholder() {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    plotsDiv.innerHTML = `
        <div class="plots-placeholder">
            <i class="fas fa-chart-bar"></i>
            <p>No plots generated</p>
        </div>
    `;
}

function clearPlots() {
    const plotsDiv = document.getElementById('plots');
    plotsDiv.className = 'plots-content';
    plotsDiv.innerHTML = `
        <div class="plots-placeholder">
            <i class="fas fa-chart-bar"></i>
            <p>Plots will appear here when you use matplotlib</p>
        </div>
    `;
}

function showOutput(content, type = 'info') {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.classList.add(`output-${type}`);
    outputDiv.innerHTML = content;
}

function clearOutput() {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.innerHTML = `
        <div class="output-placeholder">
            <i class="fas fa-info-circle"></i>
            <p>Output will appear here after running your code</p>
        </div>
    `;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
    
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.disabled = show;
        if (show) {
            runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        } else {
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run Code';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to run
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runPythonCode();
    }
    
    // Ctrl/Cmd + K to clear
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        clearEditor();
        clearOutput();
        clearPlots();
    }
});
