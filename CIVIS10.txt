import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy  # For keyword extraction

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")  # Load spaCy model for NLP tasks

# Function to extract keywords using spaCy
def extract_keywords(content):
    doc = nlp(content)
    keywords = []

    # Extracting the most frequent nouns and named entities
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:  # Nouns and proper nouns are potential keywords
            keywords.append(token.text.lower())
    
    # Remove duplicates and return the top 10 keywords
    unique_keywords = list(set(keywords))
    return unique_keywords[:10]  # Limit to top 10 keywords

# Function to assess Currency
def assess_currency(pub_date):
    current_year = datetime.now().year
    publication_year = datetime.strptime(pub_date, "%Y-%m-%d").year
    age = current_year - publication_year
    if age <= 2:
        return "The information is very recent and up-to-date."
    elif age <= 5:
        return "The information is fairly recent, but may not reflect the latest developments."
    else:
        return "The information is outdated and may not reflect current knowledge."

# Function to assess Relevance
def assess_relevance(keywords, content):
    matched_keywords = [word for word in keywords if word.lower() in content.lower()]
    if matched_keywords:
        return f"The content is relevant as it discusses the following keywords: {', '.join(matched_keywords)}."
    else:
        return "The content may not be directly relevant to your needs as it doesn't mention the specified keywords."

# Function to assess Authority
def assess_authority(author, domain):
    trusted_domains = ["edu", "gov", "org"]
    if any(domain.endswith(trusted) for trusted in trusted_domains):
        return "The source is credible due to the presence of a trusted domain (e.g., .edu, .gov, .org)."
    elif author and "PhD" in author:
        return f"The author appears authoritative, likely a subject matter expert, as they hold a PhD: {author}."
    else:
        return "The source's authority is unclear. The author may not be a recognized expert or the domain is not well-known."

# Function to assess Accuracy
def assess_accuracy(content):
    if "fact-checked" in content:
        return "The content appears accurate, with fact-checking indicators present."
    else:
        return "The accuracy of the content cannot be confirmed. It lacks clear verification from trusted sources."

# Function to assess Purpose
def assess_purpose(content):
    if "advertisement" in content or "buy" in content:
        return "The content seems to have a commercial or persuasive purpose, which may introduce bias."
    elif "inform" in content:
        return "The content appears to be informative and objective."
    else:
        return "The purpose of the content is unclear. It may have an entertainment or neutral tone."

# Function to perform Sentiment and Bias Analysis
def analyze_sentiment_and_bias(content):
    sentiment_score = sentiment_analyzer.polarity_scores(content)
    sentiment = "neutral"
    if sentiment_score['compound'] >= 0.05:
        sentiment = "positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "negative"

    # Bias detection using TextBlob
    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        bias = "positive bias detected"
    elif polarity < -0.1:
        bias = "negative bias detected"
    else:
        bias = "neutral content, no clear bias detected"
    
    return sentiment, bias

# Function to perform semantic search to find relevant documents
def semantic_search(query, document_list):
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    document_embeddings = sentence_model.encode(document_list, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    top_results = similarities[0].topk(5)
    results = [document_list[i] for i in top_results[1]]
    return results

# Function to scrape article content and metadata
def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = ' '.join([p.text for p in soup.find_all('p')])
    
    pub_date = soup.find("meta", {"name": "date"})["content"] if soup.find("meta", {"name": "date"}) else "Unknown"
    author = soup.find("meta", {"name": "author"})["content"] if soup.find("meta", {"name": "author"}) else "Unknown"
    domain = url.split("/")[2]
    
    return content, pub_date, author, domain

# Main function to evaluate the webpage and return analysis
def evaluate_page(url):
    # Scrape the article
    content, pub_date, author, domain = scrape_article(url)
    
    # Extract keywords from the content
    keywords = extract_keywords(content)
    
    # Perform CRAAP test using extracted keywords
    currency_analysis = assess_currency(pub_date)
    relevance_analysis = assess_relevance(keywords, content)
    authority_analysis = assess_authority(author, domain)
    accuracy_analysis = assess_accuracy(content)
    purpose_analysis = assess_purpose(content)
    
    # Perform sentiment and bias analysis
    sentiment, bias = analyze_sentiment_and_bias(content)
    
    # Combine the analyses to form the final report
    analysis_report = {
        "Currency": currency_analysis,
        "Relevance": relevance_analysis,
        "Authority": authority_analysis,
        "Accuracy": accuracy_analysis,
        "Purpose": purpose_analysis,
        "Sentiment": f"The overall sentiment of the content is: {sentiment}.",
        "Bias": bias,
    }
    
    return analysis_report

# Example usage
url = "https://www.nytimes.com/2025/02/04/us/politics/trump-gaza-strip-netanyahu.html"

analysis = evaluate_page(url)

# Print the analysis report
for criterion, analysis_text in analysis.items():
    print(f"{criterion}: {analysis_text}\n")
