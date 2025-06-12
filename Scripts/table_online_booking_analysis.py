import pandas as pd

# Load the dataset
df = pd.read_csv("Data\Dataset .csv")

# Fill missing values if needed
df['Cuisines'].fillna('Unknown', inplace=True)

# Convert relevant columns to lowercase for consistency
df['Has Table booking'] = df['Has Table booking'].str.lower()
df['Has Online delivery'] = df['Has Online delivery'].str.lower()

# ------------------------------------------------
# 🔹 Task 1: Table Booking and Online Delivery
# ------------------------------------------------

# 1. Percentage of restaurants that offer table booking and online delivery
table_booking_pct = df['Has Table booking'].value_counts(normalize=True) * 100
online_delivery_pct = df['Has Online delivery'].value_counts(normalize=True) * 100

print("Table Booking Percentage:\n", table_booking_pct)
print("\nOnline Delivery Percentage:\n", online_delivery_pct)

# 2. Compare average ratings: table booking vs no table booking
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
avg_rating_table_booking = df.groupby('Has Table booking')['Aggregate rating'].mean()
print("\nAverage Ratings Based on Table Booking:\n", avg_rating_table_booking)

# 3. Online delivery availability by price range
delivery_by_price_range = df.groupby(['Price range', 'Has Online delivery']).size().unstack(fill_value=0)
print("\nOnline Delivery by Price Range:\n", delivery_by_price_range)

# ------------------------------------------------
# 🔹 Task 2: Price Range Analysis
# ------------------------------------------------

# 1. Most common price range
most_common_price = df['Price range'].mode()[0]
print("\nMost Common Price Range:", most_common_price)

# 2. Average rating for each price range
avg_rating_per_price = df.groupby('Price range')['Aggregate rating'].mean()
print("\nAverage Rating per Price Range:\n", avg_rating_per_price)

# 3. Rating color with highest average rating
rating_color_avg = df.groupby('Rating color')['Aggregate rating'].mean()
highest_rated_color = rating_color_avg.idxmax()
print("\nRating Color with Highest Average Rating:", highest_rated_color)

# ------------------------------------------------
# 🔹 Task 3: Feature Engineering
# ------------------------------------------------

# 1. Extract length-based features
df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)
df['Address Length'] = df['Address'].apply(len)

# 2. Binary encoding for 'Has Table booking' and 'Has Online delivery'
df['Has_Table_Booking_Binary'] = df['Has Table booking'].apply(lambda x: 'yes' if x == 'yes' else 'no')
df['Has_Online_Delivery_Binary'] = df['Has Online delivery'].apply(lambda x: 'yes' if x == 'yes' else 'no')

# Show sample of new features
print("\nSample Engineered Features:\n", df[['Restaurant Name', 'Restaurant Name Length', 'Address', 'Address Length', 'Has_Table_Booking_Binary', 'Has_Online_Delivery_Binary']].head())
