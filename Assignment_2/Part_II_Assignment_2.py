#=========
# Part 2
#=========

### Code Availability and Utilization Notes

# The code provided in this repository has been optimized with the assistance of GPT-4 and Claude 3 Sonnet, showcasing the application of cutting-edge AI models in enhancing software development and data analysis methodologies. 

### On Using OpenAI's API

# I attempted to integrate OpenAI's API in this project. However, I encountered challenges due to recent updates to the API since my last use. These changes have left me uncertain about the best approach to integrate it effectively into my current project workflow. I welcome suggestions or guidance on adapting to the latest API changes for future iterations of this project.


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import openai
openai.api_key = '####'

# Ensure necessary NLTK resources are available
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to clean and tokenize text for frequency analysis
def clean_and_tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    return words

# Function to calculate word frequencies
def calculate_word_frequencies(words):
    frequencies = {}
    for word in words:
        if word not in frequencies:
            frequencies[word] = 1
        else:
            frequencies[word] += 1
    return frequencies

# Function to split text into individual reviews
def split_into_reviews(text):
    reviews = []
    for review in text.split("Review #"):
        if review.strip():
            review_num = review.split(":")[0].strip()
            review_text = ":".join(review.split(":")[1:]).strip()
            reviews.append((review_num, review_text))
    return reviews  # No need to skip the first review

# Function for TF-IDF calculation across all reviews
def calculate_tfidf(reviews):
    review_texts = [review_text for _, review_text in reviews]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(review_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0)
    scores_dict = {feature_names[i]: tfidf_scores[0, i] for i in range(tfidf_matrix.shape[1])}
    return scores_dict

# Sentiment analysis with review numbering
def sentiment_analysis_with_numbers(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for review_num, review_text in reviews:
        score = sia.polarity_scores(review_text)
        sentiment_scores.append((review_num, score))
    return sentiment_scores

# Main analysis function
def visualize_word_frequencies(frequencies):
    plt.figure(figsize=(10, 6))
    top_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:10])
    words, counts = zip(*top_frequencies.items())
    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.title("Top 10 Words by Frequency")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Function to visualize TF-IDF scores
def visualize_tfidf_scores(tfidf_scores):
    plt.figure(figsize=(10, 6))
    top_tfidf_scores = dict(sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:10])
    words, scores = zip(*top_tfidf_scores.items())
    plt.bar(words, scores)
    plt.xticks(rotation=90)
    plt.title("Top 10 Words by TF-IDF Score")
    plt.xlabel("Words")
    plt.ylabel("TF-IDF Score")
    plt.tight_layout()
    plt.show()

# Function to visualize sentiment scores
def visualize_sentiment_scores(sentiment_scores_with_numbers):
    plt.figure(figsize=(10, 6))
    review_numbers, scores = zip(*[(int(num), score['compound']) for num, score in sentiment_scores_with_numbers])
    plt.bar(review_numbers, scores)
    plt.xticks(review_numbers)
    plt.title("Sentiment Scores")
    plt.xlabel("Review Number")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.show()

# Main analysis function
def main():
    text = read_text_file('movie_reviews.txt')
    reviews = split_into_reviews(text)
    words = clean_and_tokenize(text)

    frequencies = calculate_word_frequencies(words)
    tfidf_scores = calculate_tfidf(reviews)
    sentiment_scores_with_numbers = sentiment_analysis_with_numbers(reviews)

    # Output top words by frequency and TF-IDF
    print("Top words by frequency:", sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:10])
    print("Top words by TF-IDF:", sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:10])

    # Output sentiment scores for all reviews
    for review_num, score in sentiment_scores_with_numbers:
        print(f"\nReview #{review_num}: {score}")

    # Visualize top 10 word frequencies
    visualize_word_frequencies(frequencies)

    # Visualize top 10 TF-IDF scores
    visualize_tfidf_scores(tfidf_scores)

    # Visualize sentiment scores
    visualize_sentiment_scores(sentiment_scores_with_numbers)

# def classify_review_gpt(review):
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=f"Classify the following movie review into the categories of Originality, Entertainment Value, Character Development, and Themes and Messages:\n\n'{review}'\n\nCategory:",
#         temperature=0.7,
#         max_tokens=64,
#         n=1,
#         stop=["\n"]
#     )
#     category = response['choices'][0]['text'].strip()
#     return category


# def main():
#     text = read_text_file('movie_reviews.txt')
#     reviews = split_into_reviews(text)

#     categorized_reviews = {
#         "Originality": [],
#         "Entertainment Value": [],
#         "Character Development": [],
#         "Themes and Messages": []
#     }

#     for review_num, review_text in reviews:
#         category = classify_review_gpt(review_text)
#         if category in categorized_reviews:
#             categorized_reviews[category].append((review_num, review_text))
#         else:
#             print(f"Uncategorized review #{review_num}, category detected: '{category}'")

#     # Output categorized reviews
#     for category, reviews in categorized_reviews.items():
#         print(f"\nCategory: {category} ({len(reviews)} Reviews):")
#         for review_num, review_text in reviews:
#             print(f"\nReview #{review_num}: {review_text[:100]}...")  # Shows a snippet for brevity
            
if __name__ == "__main__":
    main()