import os
import pandas as pd
from datetime import datetime

def label_article(title, description):
    print("\nTitle:", title)
    print("Description:", description)
    while True:
        label = input("Enter label (1 or 0): ")
        if label in {'1', '0'}:
            return int(label)
        else:
            print("Invalid input. Please enter 1 or 0.")

def label_articles(df):
    for index, row in df.iterrows():
        if pd.isna(row['label']):
            df.at[index, 'label'] = label_article(row['title'], row['description'])
    return df

# Load the articles DataFrame
archive_dir = '/Users/jakepappo/Documents/Stuff/Projects/news_agg/archive/'
csv_filename = input("Enter the name of the CSV file (without the path): ")
csv_filepath = os.path.join(archive_dir, csv_filename)

try:
    articles_df = pd.read_csv(csv_filepath)
except FileNotFoundError:
    print(f"File {csv_filepath} not found.")
    exit(1)

# Label the articles
articles_df = label_articles(articles_df)

# Save the updated DataFrame
today_date_str = datetime.now().strftime('%m_%d_%y')
updated_archive_filepath = os.path.join(archive_dir, f'articles_all_{today_date_str}.csv')
articles_df.to_csv(updated_archive_filepath, index=False)
print(f"Updated file saved to {updated_archive_filepath}")