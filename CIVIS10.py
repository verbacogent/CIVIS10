import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
import argparse

# Initialize models (move these outside the functions for efficiency)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()
try:  # Handle potential spaCy model loading errors
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.  Please download it.")
    exit()  # Exit the program if the model can't be loaded

# ... (rest of the functions: extract_keywords, assess_relevance, assess_authority, assess_accuracy, assess_purpose, analyze_sentiment_and_bias, and semantic_search remain the same)

def assess_currency(pub_date):
    current_year = datetime.now().year
    try:
        publication_year = datetime.strptime(pub_date, "%Y-%m-%d").year
        age = current_year - publication_year
        if age <= 2:
            return "The information is very recent and up-to-date."
        elif age <= 5:
            return "The information is fairly recent, but may not reflect the latest developments."
        else:
            return "The information is outdated and may not reflect current knowledge."
    except ValueError:
        return "Publication date is unavailable or in an unexpected format."

def scrape_article(url):
    try:
        response = requests.get(url, timeout=10)  # Add a timeout to prevent indefinite hanging
        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        content = ' '.join([p.text for p in soup.find_all('p')])

        pub_date_tag = soup.find("meta", {"name": "date"})
        pub_date = pub_date_tag["content"] if pub_date_tag and pub_date_tag.has_attr('content') else "Unknown" # Check if the tag exists AND has the 'content' attribute.

        author_tag = soup.find("meta", {"name": "author"})
        author = author_tag["content"] if author_tag and author_tag.has_attr('content') else "Unknown" # Check if the tag exists AND has the 'content' attribute.

        domain = url.split("/")[2] if len(url.split("/")) > 2 else "Unknown"

        return content, pub_date, author, domain

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None, None, None, None

    except Exception as e:  # Catch other potential BeautifulSoup errors
        print(f"Error parsing HTML: {e}")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Analyze webpage content.")
    parser.add_argument("url", help="The URL of the webpage to analyze.")
    args = parser.parse_args()
    url = args.url

    try:
        print("Starting CIVIS10 algorithm...")

        content, pub_date, author, domain = scrape_article(url)

        if content is None:
            print("Scraping failed. Exiting.")
            return

        keywords = extract_keywords(content)

        currency_analysis = assess_currency(pub_date)
        relevance_analysis = assess_relevance(keywords, content)
        authority_analysis = assess_authority(author, domain)
        accuracy_analysis = assess_accuracy(content)
        purpose_analysis = assess_purpose(content)

        sentiment, bias = analyze_sentiment_and_bias(content)

        analysis_report = {
            "Currency": currency_analysis,
            "Relevance": relevance_analysis,
            "Authority": authority_analysis,
            "Accuracy": accuracy_analysis,
            "Purpose": purpose_analysis,
            "Sentiment": f"The overall sentiment of the content is: {sentiment}.",
            "Bias": bias,
        }

        print("Analysis complete! Final report:")
        print(analysis_report)

        print("CIVIS10 algorithm finished.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
