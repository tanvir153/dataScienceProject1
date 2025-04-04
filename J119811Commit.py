# Importing all necessary libraries
import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import nltk
import seaborn as sns
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Ensure NLTK resources are downloaded
nltk.download('stopwords')  
nltk.download('punkt')  

# Function to load and parse JSON data into a DataFrame
def load_data(file_path):
    """
    Reads a JSON file and converts it into a Pandas DataFrame.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        pd.DataFrame: A DataFrame containing the JSON data.
    """
    print("Loading dataset...")  
    try:
        data = pd.read_json(file_path, lines=True)
        print("Dataset loaded successfully!")
        return data
    except ValueError as e:
        print(f"Error loading JSON file: {e}")
        return pd.DataFrame()

# Function to create visualizations
def create_visualizations(data, asin):
    """
    Generate visualizations for a given product dataset.
    Args:
        data (DataFrame): Product-specific data.
        asin (str): Product ID.
    """
    # Bar Plot for Ratings Distribution
    plt.figure(figsize=(10, 6))  
    sns.countplot(data=data, x='overall', palette='coolwarm')
    plt.title(f"Ratings Distribution for Product {asin}", fontsize=14)
    plt.xlabel("Ratings")
    plt.ylabel("Count")
    plt.savefig(f'ratings_distribution_{asin}.png')
    plt.show()

    # Word Cloud for Reviews
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    all_words = " ".join(data['reviewText'])
    tokens = [stemmer.stem(word) for word in tokenizer.tokenize(all_words.lower()) if word not in stop_words]
    word_freq = Counter(tokens)

    # Introducing variation by altering word frequencies slightly
    for key in word_freq.keys():
        word_freq[key] += 1  # Adding a constant to all frequencies for randomness

    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Product {asin}", fontsize=14)
    plt.savefig(f'wordcloud_{asin}.png')
    plt.show()

    # Sentiment Distribution for the profucts (1st commit as comment)
    plt.figure(figsize=(10, 6))  
    sns.histplot(data['sentiment'], kde=True, color='purple', bins=20)
    plt.title(f"Sentiment Analysis for Product {asin}", fontsize=14)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.savefig(f'sentiment_distribution_{asin}.png')
    plt.show()

    # Sentiment Trend Over Time
    data['reviewTime'] = pd.to_datetime(data['reviewTime'])
    monthly_sentiment = data.groupby(data['reviewTime'].dt.to_period("M"))['sentiment'].mean()
    monthly_sentiment.plot(kind='line', marker='o', figsize=(10, 6), color='orange')  # pandas plot function
    plt.title(f"Sentiment Trend Over Time for Product {asin}")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.grid(True)
    plt.savefig(f'sentiment_trend_{asin}.png')
    plt.show()

    # Bubble Chart: Ratings vs Votes
    print(f"Creating Bubble Chart for {asin}...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data['vote'], 
        y=data['overall'], 
        size=data['vote'], 
        sizes=(50, 500), 
        hue=data['overall'], 
        palette='cool',    # color and size change of bubble chart another commit
        alpha=0.7
    )
    plt.title(f"Bubble Chart: Ratings vs Votes for Product {asin}", fontsize=14)
    plt.xlabel("Votes")
    plt.ylabel("Ratings")
    plt.legend(title="Ratings", loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(f'bubble_chart_{asin}.png')
    plt.show()

    # Interactive Dashboard with Plotly
    print("Creating interactive dashboard...")  
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ratings vs Votes", "Sentiment Distribution"))

    fig.add_trace(
        go.Scatter(x=data['vote'], y=data['overall'], mode='markers',
                   marker=dict(
            size=data['vote'],  # Bubble size directly proportional to votes
            color=data['overall'],  # Used ratings for color
            colorscale='Cividis',  # Updated color scale
            showscale=True),
                   name="Votes vs Ratings"),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(x=data['sentiment'], nbinsx=20, marker_color='teal', name="Sentiment"),
        row=1, col=2
    )

    fig.update_layout(height=600, width=1000, title_text=f"Dashboard for Product {asin}")
    fig.write_html(f'interactive_dashboard_{asin}.html')
    fig.show()

# File path to the dataset
file_path = "D:\\Data Science Project\\Industrial_and_Scientific.json"

# Load dataset
reviews_df = load_data(file_path)

# Data Cleaning and Preprocessing
print("Data cleaning and preprocessing started...")
columns_to_keep = ['asin', 'reviewText', 'overall', 'unixReviewTime', 'summary', 'reviewerName', 'vote']
reviews_df = reviews_df[columns_to_keep]

# Handle missing values and clean the 'vote' column
reviews_df['reviewerName'] = reviews_df['reviewerName'].fillna("Anonymous")
reviews_df['vote'] = reviews_df['vote'].fillna(0).astype(str).str.replace(',', '').astype(int)
reviews_df.dropna(subset=['reviewText', 'overall'], inplace=True)

# Convert UNIX timestamp to readable date
reviews_df['reviewTime'] = pd.to_datetime(reviews_df['unixReviewTime'], unit='s')
print("Data cleaning completed!")

# Analyze and Visualize Data
products = reviews_df['asin'].unique()[3:6]  # Selecting a unique range of three products for analysis
product_data = {asin: reviews_df[reviews_df['asin'] == asin] for asin in products}

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer() 

for asin, data in product_data.items():
    print(f"\nAnalyzing Product: {asin}")

    # Sentiment Scores
    data['sentiment'] = data['reviewText'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Generate Visualizations
    create_visualizations(data, asin)

print("Result is saved as analysis is completed.")

# Save Cleaned Data
output_file = 'J119811_updated_with_dashboard.csv'  
reviews_df.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to {output_file}.")



# References
# Plotly. (n.d.). https://plotly.com/python/
# View of VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. (n.d.). 
#     https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399
# Waskom, M. (2021). seaborn: statistical data visualization. The Journal of Open Source Software, 6(60), 3021.
#     https://doi.org/10.21105/joss.03021
# Gallery of Examples — wordcloud 1.8.1 documentation. (n.d.). 
#     https://amueller.github.io/word_cloud/auto_examples/index.html
# MatPlotLib: a 2D Graphics environment. (2007, June 1). IEEE Journals & Magazine | IEEE Xplore. 
#     https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4160265
