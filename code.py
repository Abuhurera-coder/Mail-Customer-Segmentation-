"""
Enhanced Customer Segmentation with RFM Analysis
- Robust error handling
- Automated cluster selection
- Detailed reporting
- Cross-platform compatibility
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import joblib
import sys

# ======================
# CONFIGURATION
# ======================
class Config:
    # Get the current script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Point directly to your data folder
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Use the exact filename you have
    DATA_FILE = 'marketing_campaign.csv'  
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
    
    # Output directories
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    
    # RFM features to use
    RFM_FEATURES = ['Recency', 'Income', 'MntWines']
    
    # Model configuration
    MIN_CLUSTERS = 2  # Minimum number of clusters to try
    MAX_CLUSTERS = 8  # Maximum number of clusters to try
    N_CLUSTERS = 5    # Default number of clusters
    RANDOM_STATE = 42 # For reproducibility
    TEST_SIZE = 0.2   # For validation split

# Add verification before running
print(f"Looking for data at: {Config.DATA_PATH}")
print(f"File exists: {os.path.exists(Config.DATA_PATH)}")

# ======================
# DATA PROCESSING
# ======================
def load_data():
    """Load data with comprehensive error handling"""
    try:
        # Try tab separator first (common in marketing data)
        try:
            df = pd.read_csv(Config.DATA_PATH, sep='\t')
        except:
            # Fallback to comma separator
            df = pd.read_csv(Config.DATA_PATH)
            
        print(f"✅ Data loaded successfully from:\n{Config.DATA_PATH}")
        print(f"\nDataset Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"\n❌ Critical Error loading data: {str(e)}", file=sys.stderr)
        print("\nTroubleshooting Steps:")
        print(f"1. Verify file exists at: {os.path.abspath(Config.DATA_PATH)}")
        print("2. Check file permissions")
        print("3. Ensure file is not corrupted")
        print("4. Validate file encoding (try UTF-8)")
        sys.exit(1)

def preprocess_data(df):
    """Advanced preprocessing with validation"""
    # Validate columns
    missing_cols = [col for col in Config.RFM_FEATURES if col not in df.columns]
    if missing_cols:
        print(f"\n❌ Missing required columns: {missing_cols}", file=sys.stderr)
        print("\nAvailable columns:")
        print(df.columns.tolist())
        sys.exit(1)
    
    # Data quality report
    print("\n📊 Data Quality Report:")
    print(df[Config.RFM_FEATURES].describe())
    
    print("\n🔍 Missing Values:")
    print(df[Config.RFM_FEATURES].isna().sum())
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[Config.RFM_FEATURES] = imputer.fit_transform(df[Config.RFM_FEATURES])
    
    # Remove infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[Config.RFM_FEATURES])
    
    return X_scaled, df, scaler

# ======================
# MODEL TRAINING
# ======================
def find_optimal_clusters(X_scaled):
    """Automated cluster selection using silhouette score"""
    silhouette_scores = []
    
    for k in range(Config.MIN_CLUSTERS, Config.MAX_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(Config.MIN_CLUSTERS, Config.MAX_CLUSTERS + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Optimal Cluster Selection')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()
    
    optimal_k = np.argmax(silhouette_scores) + Config.MIN_CLUSTERS
    print(f"\n🎯 Optimal number of clusters: {optimal_k}")
    return optimal_k

def train_model(X_scaled):
    """Train model with optimal clusters"""
    optimal_k = find_optimal_clusters(X_scaled)
    
    model = KMeans(n_clusters=optimal_k,
                  random_state=Config.RANDOM_STATE,
                  n_init=10)  # Explicitly set n_init to avoid warning
    
    model.fit(X_scaled)
    
    # Save model artifacts
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.MODEL_DIR, 'rfm_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model

# ======================
# VISUALIZATION & REPORTING
# ======================
def visualize_results(df, model):
    """Enhanced visualization with cluster profiles"""
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Recency vs Income
    plt.subplot(1, 3, 1)
    plt.scatter(df['Recency'], df['Income'], c=model.labels_, cmap='viridis', alpha=0.6)
    plt.xlabel('Recency (days)')
    plt.ylabel('Income')
    plt.title('Recency vs Income')
    
    # Plot 2: Income vs Spending
    plt.subplot(1, 3, 2)
    plt.scatter(df['Income'], df['MntWines'], c=model.labels_, cmap='viridis', alpha=0.6)
    plt.xlabel('Income')
    plt.ylabel('Wine Spending')
    plt.title('Income vs Spending')
    
    # Plot 3: Recency vs Spending
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(df['Recency'], df['MntWines'], c=model.labels_, cmap='viridis', alpha=0.6)
    plt.xlabel('Recency')
    plt.ylabel('Wine Spending')
    plt.title('Recency vs Spending')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_DIR, 'cluster_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {plot_path}")
    plt.show()

def generate_report(df, model):
    """Comprehensive cluster analysis report"""
    report = df.groupby('Cluster')[Config.RFM_FEATURES].agg(['mean', 'median', 'count'])
    
    print("\n📈 Cluster Analysis Report:")
    print(report.round(2))
    
    # Save full report
    report_path = os.path.join(Config.OUTPUT_DIR, 'cluster_report.csv')
    report.to_csv(report_path)
    print(f"\n📄 Full report saved to: {report_path}")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Setup directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    print("="*60)
    print("🎯 CUSTOMER SEGMENTATION ANALYSIS")
    print("="*60)
    
    try:
        # 1. Load data
        print("\n📂 Loading data...")
        df = load_data()
        
        # 2. Preprocess
        print("\n⚙️ Preprocessing data...")
        X_scaled, df, scaler = preprocess_data(df)
        
        # 3. Train model
        print("\n🤖 Training model...")
        model = train_model(X_scaled)
        df['Cluster'] = model.labels_
        
        # 4. Save results
        output_path = os.path.join(Config.OUTPUT_DIR, 'segmented_customers.csv')
        df.to_csv(output_path, index=False)
        print(f"\n💾 Segmented data saved to: {output_path}")
        
        # 5. Visualize and report
        print("\n📊 Generating insights...")
        visualize_results(df, model)
        generate_report(df, model)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)