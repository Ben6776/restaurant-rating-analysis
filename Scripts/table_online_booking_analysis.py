import pandas as pd

df = pd.read_csv("Data\Dataset .csv")

df['Cuisines'].fillna('Unknown', inplace=True)

df['Has Table booking'] = df['Has Table booking'].str.lower()
df['Has Online delivery'] = df['Has Online delivery'].str.lower()

table_booking_pct = df['Has Table booking'].value_counts(normalize=True) * 100
online_delivery_pct = df['Has Online delivery'].value_counts(normalize=True) * 100

print("Table Booking Percentage:\n", table_booking_pct)
print("\nOnline Delivery Percentage:\n", online_delivery_pct)

df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
avg_rating_table_booking = df.groupby('Has Table booking')['Aggregate rating'].mean()
print("\nAverage Ratings Based on Table Booking:\n", avg_rating_table_booking)

delivery_by_price_range = df.groupby(['Price range', 'Has Online delivery']).size().unstack(fill_value=0)
print("\nOnline Delivery by Price Range:\n", delivery_by_price_range)


most_common_price = df['Price range'].mode()[0]
print("\nMost Common Price Range:", most_common_price)

avg_rating_per_price = df.groupby('Price range')['Aggregate rating'].mean()
print("\nAverage Rating per Price Range:\n", avg_rating_per_price)

rating_color_avg = df.groupby('Rating color')['Aggregate rating'].mean()
highest_rated_color = rating_color_avg.idxmax()
print("\nRating Color with Highest Average Rating:", highest_rated_color)


df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)
df['Address Length'] = df['Address'].apply(len)

df['Has_Table_Booking_Binary'] = df['Has Table booking'].apply(lambda x: 'yes' if x == 'yes' else 'no')
df['Has_Online_Delivery_Binary'] = df['Has Online delivery'].apply(lambda x: 'yes' if x == 'yes' else 'no')

print("\nSample Engineered Features:\n", df[['Restaurant Name', 'Restaurant Name Length', 'Address', 'Address Length', 'Has_Table_Booking_Binary', 'Has_Online_Delivery_Binary']].head())
