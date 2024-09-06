import os
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define the RSS feeds
feeds = {
    'WSJ_Opinion': 'https://feeds.a.dj.com/rss/RSSOpinion.xml',
    'WSJ_WorldNews': 'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
    'WSJ_USBussiness': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
    'WSJ_MarketsMain': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'WSJ_Tech': 'https://feeds.a.dj.com/rss/RSSWSJD.xml',
    'WSJ_Lifestyle': 'https://feeds.a.dj.com/rss/RSSLifestyle.xml',
    'NYT_World': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    'NYT_US': 'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
    'NYT_Politics': 'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml',
    'NYT_Business': 'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'NYT_Economy': 'https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml',
    'NYT_EnergyEnvironment': 'https://rss.nytimes.com/services/xml/rss/nyt/EnergyEnvironment.xml',
    'NYT_Tech': 'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'NYT_PersonalTech': 'https://rss.nytimes.com/services/xml/rss/nyt/PersonalTech.xml',
    'NYT_Sports': 'https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml',
    'NYT_Science': 'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml',
    'NYT_Space': 'https://rss.nytimes.com/services/xml/rss/nyt/Space.xml',
    'NYT_BooksReview': 'https://rss.nytimes.com/services/xml/rss/nyt/Books/Review.xml',
    'NYT_Movies': 'https://rss.nytimes.com/services/xml/rss/nyt/Movies.xml',
    'FT': 'https://www.ft.com/myft/following/1e8ea6d7-389e-4fbe-89e1-e0dd4426afee.rss'
}

def parse_rss(feed_url, feed_name, is_ft=False, is_wsj=False):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        # Extract the title and description
        title = entry.title
        description = entry.description
        # Strip HTML tags for FT descriptions
        if is_ft:
            soup = BeautifulSoup(description, 'html.parser')
            for tag in soup(['a', 'img', 'br']):
                tag.decompose()
            description = soup.get_text(separator=" ").strip().rstrip(' ,')

        # Extract the publication date
        date = None
        if 'published' in entry:
            date = entry.published
        elif 'published_parsed' in entry:
            date = datetime(*entry.published_parsed[:6]).isoformat()
        if date:
            date = date.replace('GMT', '+0000')

        # Extract categories
        categories = []
        if 'tags' in entry:
            for tag in entry.tags:
                if is_wsj and tag.get('term') == 'PAID':
                    continue
                categories.append(tag.get('term').lower())

        # Add all info to the article
        link = entry.link
        articles.append({
            'title': title,
            'description': description,
            'categories': categories,
            'date': date,
            'link': link,
            'feed_name': feed_name,
            'source': feed_name.split('_')[0],
        })
    return articles

# Collect articles from all feeds into a list of dictionaries
all_articles_list = []
for feed_name, feed_url in feeds.items():
    is_ft = 'FT' in feed_name
    is_wsj = 'WSJ' in feed_name
    articles = parse_rss(feed_url, feed_name, is_ft=is_ft, is_wsj=is_wsj)
    all_articles_list.extend(articles)

# Convert the list of dictionaries to a DataFrame
new_articles_df = pd.DataFrame(all_articles_list)

# Merge the current articles into the aggregate list
def update_articles_archive(archive_df, current_df):
    # Create a unique identifier based on the title
    archive_df['identifier'] = archive_df['title']
    current_df['identifier'] = current_df['title']

    # Create a dictionary from the existing articles to preserve labels
    if 'label' in archive_df.columns:
        existing_articles = archive_df.set_index('identifier')['label'].to_dict()
    else:
        existing_articles = {}

    # Update or add new articles, preserving labels if they exist
    for i, row in current_df.iterrows():
        identifier = row['identifier']
        if identifier in existing_articles:
            current_df.at[i, 'label'] = existing_articles[identifier]

    # Combine and remove duplicates
    combined_df = pd.concat([archive_df, current_df]).drop_duplicates(subset='identifier', keep='last')
    combined_df = combined_df.drop(columns=['identifier'])
    
    return combined_df

# Define the path to the archive directory
archive_dir = '/Users/jakepappo/Documents/Stuff/Projects/news_agg/archive/'

# Load yesterday's articles DataFrame
yesterday_date_str = (datetime.now() - pd.Timedelta(days=1)).strftime('%m_%d_%y')
archive_filepath = os.path.join(archive_dir, f'articles_all_{yesterday_date_str}.csv')

try:
    archive_df = pd.read_csv(archive_filepath)
except FileNotFoundError:
    print(f"File {archive_filepath} not found. Starting with an empty DataFrame.")
    archive_df = pd.DataFrame()

# Update the articles DataFrame with new data
articles_df = update_articles_archive(archive_df, new_articles_df)

# Load the BERT model and tokenizer for classification
tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")
model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")

categories = ["World", "Sports", "Business", "Sci/Tech"]

# Function to classify title+description using the fine-tuned BERT model
def classify_article(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return categories[predicted_class], probabilities[0].tolist()

def classify_row(row):
    # If the predictions have already been calculated, return them
    if pd.notna(row.get('predicted_category')) and pd.notna(row.get('predicted_probs')):
        return pd.Series([row['predicted_category'], row['predicted_probs']])
    # If not, make the predictions and store them
    title = row['title'] if pd.notna(row['title']) else ""
    description = row['description'] if pd.notna(row['description']) else ""
    text = f"{title} - {description}"
    predicted_category, probabilities = classify_article(text)
    return pd.Series([predicted_category, probabilities])

# Apply classification to the articles
articles_df[['predicted_category', 'predicted_probs']] = articles_df.apply(classify_row, axis=1)

# Define the path for today's updated archive
today_date_str = datetime.now().strftime('%m_%d_%y')
updated_archive_filepath = os.path.join(archive_dir, f'articles_all_{today_date_str}.csv')

# Save the updated DataFrame
articles_df.to_csv(updated_archive_filepath, index=False)