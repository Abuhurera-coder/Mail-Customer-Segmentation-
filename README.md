Customer Segmentation Project

Customer Segmentation Visualization

📌 Table of Contents
Project Overview

Features

Installation Guide

Usage

Project Structure

Customization

Outputs & Results

Troubleshooting

Contributing

License

Contact

🔍 Project Overview
This project performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering to identify distinct customer groups for targeted marketing strategies.

Key Objectives
✔ Cluster customers based on purchasing behavior
✔ Visualize segments for better decision-making
✔ Generate reports for marketing teams

✨ Features
✅ Automated RFM Calculation - Converts raw transaction data into RFM metrics
✅ Optimal Cluster Selection - Uses Silhouette Score to determine the best number of clusters
✅ Interactive Visualizations - Plots customer segments for easy interpretation
✅ Model Persistence - Saves trained models for future use
✅ Detailed Reports - Exports segmentation results in CSV and JSON formats

🛠 Installation Guide
Prerequisites
Python 3.8+

Git (optional)

Step-by-Step Setup
Clone the Repository

bash
Copy
git clone https://github.com/Abuhurera-coder/Mail-Customer-Segmentation-.git
cd Mail-Customer-Segmentation-
Set Up a Virtual Environment (Recommended)

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
pip install -r requirements.txt
Prepare Your Data

Place your dataset (customers.csv or marketing_campaign.csv) in the data/ folder.

Ensure it contains the required columns:

CustomerID

PurchaseDate

Amount

🚀 Usage
Running the Segmentation
bash
Copy
python code.py
Expected Outputs
output/cluster_visualization.png → Segmentation plots

output/segmented_customers.csv → Customers with assigned clusters

models/rfm_model_*.joblib → Saved trained models

📂 Project Structure
Copy
Mail-Customer-Segmentation/
├── data/                   # Input datasets
│   ├── customers.csv       # Sample customer data
│   └── marketing_campaign.csv  # Marketing data
├── models/                 # Saved ML models
│   └── rfm_model_*.joblib  
├── output/                 # Results & visualizations
│   ├── cluster_visualization.png
│   └── segmented_customers.csv
├── code.py                 # Main application
└── requirements.txt        # Python dependencies
⚙ Customization
Modify code.py to adjust:

RFM Features (e.g., Recency, Income, MntWines)

Cluster Range (MIN_CLUSTERS, MAX_CLUSTERS)

Visualization Style (COLOR_PALETTE, PLOT_STYLE)

📊 Outputs & Results
File	Description
cluster_visualization.png	Visual representation of customer segments
segmented_customers.csv	Customer data with assigned clusters
rfm_model_*.joblib	Saved model for future predictions
🔧 Troubleshooting
Common Issues & Fixes
❌ "Data file not found"
→ Ensure customers.csv is in the data/ folder.

❌ "Missing required columns"
→ Verify your dataset contains CustomerID, PurchaseDate, and Amount.

❌ Git push rejected
→ Run:
bash git pull origin main --allow-unrelated-histories git push -u origin main

🤝 Contributing
Fork the repository

Create a new branch (git checkout -b feature/NewFeature)

Commit changes (git commit -m "Add NewFeature")

Push (git push origin feature/NewFeature)

Open a Pull Request

📜 License
This project is licensed under MIT License.
