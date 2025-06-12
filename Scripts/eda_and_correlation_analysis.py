# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Load the dataset
df = pd.read_csv(r"Data\Dataset .csv")

# -------------------------------------
# Task 1: Data Exploration & Preprocessing
# -------------------------------------

# 1. Shape of the dataset
print("Shape of dataset:", df.shape)

# 2. Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values (only 'Cuisines' had missing values)
df['Cuisines'].fillna('Unknown', inplace=True)

# 3. Data type conversion
df['Country Code'] = df['Country Code'].astype(str)
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')

# 4. Target variable analysis - 'Aggregate rating'
print("\nAggregate Rating Distribution:\n", df['Aggregate rating'].value_counts())
plt.figure(figsize=(8,5))
sns.countplot(x='Aggregate rating', data=df, order=sorted(df['Aggregate rating'].unique()))
plt.title('Distribution of Aggregate Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------
# Task 2: Descriptive Analysis
# -------------------------------------

# 1. Basic statistics
print("\nStatistical Summary:\n", df.describe())

# 2. Categorical distributions
print("\nCountry Code Distribution:\n", df['Country Code'].value_counts())
print("\nTop Cities:\n", df['City'].value_counts().head(10))
print("\nTop Cuisines:\n", df['Cuisines'].value_counts().head(10))

# Plot top cuisines
plt.figure(figsize=(10,5))
df['Cuisines'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Cuisines")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot top cities
plt.figure(figsize=(10,5))
df['City'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Cities with Most Restaurants")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------
# Task 3: Geospatial Analysis
# -------------------------------------

# 1. Map of restaurant locations
map_restaurants = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=2)

for _, row in df.iterrows():
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=1,
            popup=row['Restaurant Name'], parse_html=True,
            color='blue',
            fill=True
        ).add_to(map_restaurants)

# Save map to HTML
map_restaurants.save("restaurant_map.html")
print("\n🗺️ Restaurant location map saved as 'restaurant_map.html'.")

# 2. Distribution of restaurants by country
plt.figure(figsize=(10,6))
df['Country Code'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Country Codes by Restaurant Count")
plt.xlabel("Country Code")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.show()

correlation = df.corr(numeric_only=True)

# 2. Unstack and remove self-pairs
cor_pairs = correlation.unstack()
cor_pairs = cor_pairs[cor_pairs.index.get_level_values(0) != cor_pairs.index.get_level_values(1)]

# 3. Remove duplicate pairs using frozenset
unique_pairs = {}
for (a, b), value in cor_pairs.items():
    key = frozenset((a, b))
    if key not in unique_pairs:
        unique_pairs[key] = value

# 4. Create sorted DataFrame
sorted_cor = pd.DataFrame([(list(k)[0], list(k)[1], v) for k, v in unique_pairs.items()],
                          columns=['Feature 1', 'Feature 2', 'Correlation'])
sorted_cor['Abs Correlation'] = sorted_cor['Correlation'].abs()
sorted_cor = sorted_cor.sort_values(by='Abs Correlation', ascending=False)

# 5. Show top 10 correlations
print("🔝 Top Correlated Feature Pairs:")
print(sorted_cor.head(10))

# 6. Get the strongest correlated pair
top_pair = sorted_cor.iloc[0]
feature_x = top_pair['Feature 1']
feature_y = top_pair['Feature 2']
print(f"\n📌 Strongest correlation is between '{feature_x}' and '{feature_y}' with value: {top_pair['Correlation']:.2f}")

# 7. Plot scatter plot for best pair
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x=feature_x, y=feature_y)
plt.title(f"Scatter Plot: {feature_x} vs {feature_y}")
plt.tight_layout()
plt.show()

# 8. Show correlation heatmap for location and ratings
location_corr = df[['Latitude', 'Longitude', 'Aggregate rating']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(location_corr, annot=True, cmap='coolwarm')
plt.title("Correlation: Latitude/Longitude vs Rating")
plt.tight_layout()
plt.show()