# 🍽️ Restaurant Rating Analysis

This project performs exploratory data analysis (EDA), feature engineering, and predictive modeling on restaurant data to understand the factors that affect restaurant ratings. The goal is to analyze features like table booking availability, online delivery options, price range, votes, and more to predict the aggregate restaurant rating.

## ✅ Tasks Overview

### EDA & Correlation
- Data cleaning and formatting
- Feature extraction (length of name, address, etc.)
- Correlation matrix heatmap
- Aggregated rating visualizations

### Booking Analysis
- Impact of table booking and online delivery on ratings
- Normalized count distribution
- Binary encoding of booking features

### Model Building
- One-hot encoding for categorical variables
- Train-test split
- Model comparison using Linear Regression, Decision Tree, Random Forest
- Evaluation metrics: R² and Mean Squared Error

## 📊 Sample Outputs

- Average rating by price range
- Booking preference impact on rating
- Most accurate model for predicting ratings

## 🌐 Interactive Map

`restaurant_map.html` visualizes the geographical locations of restaurants on an interactive map using Folium.

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ben6776/restaurant-rating-analysis.git
   cd restaurant-rating-analysis
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

## 🧠 Technologies Used

* Python 3.13+
* Pandas
* NumPy
* Seaborn & Matplotlib
* Scikit-learn
* Folium
* LazyPredict (optional for auto model comparison)

## ✍️ Author

Ben

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

