import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Dataset .csv")

# Preprocessing
df['Cuisines'].fillna('Unknown', inplace=True)
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
df = df[df['Aggregate rating'] > 0]  # Removing unrated rows
df['Has Table booking'] = df['Has Table booking'].str.lower().map({'yes': 1, 'no': 0})
df['Has Online delivery'] = df['Has Online delivery'].str.lower().map({'yes': 1, 'no': 0})

# Feature Engineering
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)
df['Address Length'] = df['Address'].apply(len)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Price range', 'Country Code', 'Rating color'], drop_first=True)

# --------------------------------------
# 🔹 Task 1: Predictive Modeling
# --------------------------------------

# Features & Target
features = ['Has Table booking', 'Has Online delivery', 'Votes', 'Restaurant Name Length', 'Address Length']
features += [col for col in df.columns if 'Price range_' in col or 'Country Code_' in col or 'Rating color_' in col]

X = df[features]
y = df['Aggregate rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Models to test
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Train and evaluate models
print("\nModel Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("R² Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

# --------------------------------------
# 🔹 Task 2: Customer Preference Analysis
# --------------------------------------

# Top cuisines by votes
top_cuisines_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(10)
print("\n Top Cuisines by Votes:\n", top_cuisines_votes)

# Average ratings by cuisine
avg_rating_by_cuisine = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
print("\n Cuisines with Highest Average Ratings:\n", avg_rating_by_cuisine)

# --------------------------------------
# 🔹 Task 3: Data Visualization
# --------------------------------------

plt.figure(figsize=(16, 12))

# Histogram of Ratings
plt.subplot(2, 2, 1)
sns.histplot(df['Aggregate rating'], bins=10, kde=True, color='skyblue')
plt.title("Distribution of Aggregate Ratings")

# Bar Plot - Avg Rating by Top Cuisines
plt.subplot(2, 2, 2)
avg_rating_by_cuisine.head(10).plot(kind='barh', color='orange')
plt.title("Top Cuisines by Avg Rating")
plt.xlabel("Average Rating")

# Bar Plot - Avg Rating by City
plt.subplot(2, 2, 3)
avg_rating_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
avg_rating_city.plot(kind='bar', color='green')
plt.title("Top Cities by Average Rating")

# Scatter Plot - Votes vs Rating
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Votes', y='Aggregate rating', hue='Has Online delivery', palette='coolwarm')
plt.title("Votes vs Aggregate Rating")

plt.tight_layout()
plt.show()
