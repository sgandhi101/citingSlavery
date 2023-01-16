import pandas as pd

# Read in the CSV file
df = pd.read_csv('search_results.csv', parse_dates=['decision_date'],
                 date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d'))

# Keep only the 'decision_date' column
df = df[['decision_date']]

# Separate
# df['decision_month'] = df['decision_date'].dt.month_name()
df['decision_year'] = df['decision_date'].dt.year

# Group by month and year and count the values
df = df.groupby('decision_year').size().reset_index(name='counts')

# Save the resulting DataFrame to a new CSV file
df.to_csv('new_citing_slavery.csv', index=False)

# Use the resulting data in Datawrapper
