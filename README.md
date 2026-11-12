# College Tier Segmentation and Classification

This project uses unsupervised and supervised Machine Learning techniques to group 500 engineering colleges into performance tiers and build a predictive model for categorizing new colleges.

## Project Overview
* Dataset: 500 college records (college_list.csv) with 5 numerical features:
  * Student-Faculty Ratio
  * Annual Fees (INR)
  * Placement Percentage
  * Average Package (LPA)
  * Infrastructure Score
* Data Preprocessing: Removed identifier text columns (College_Name) and standardized all numerical features using StandardScaler.
* Unsupervised Clustering:
  * Built dendrograms using SciPy (ward linkage) to visualize cluster structure and determine the optimal number of clusters (k = 3).
  * Applied AgglomerativeClustering to group colleges into 3 tiers: Mid-level, Best-level, and Emerging-level.
* Supervised Classification:
  * Used the generated cluster labels as target categories to train a RandomForestClassifier.
  * Achieved 100% accuracy on the test split and saved the trained model as College_category_model.pkl using pickle.

## Tools Used
* Language: Python
* Libraries: Pandas, Scikit-Learn, SciPy, Matplotlib, Pickle
