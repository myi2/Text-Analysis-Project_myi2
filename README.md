# Text-Analysis-Project: Insightful Movie Reviews - NLP Analysis of Movie Critiques
 
Please read the [instructions](instructions.md).

In this project, I focused on analyzing a dataset of movie reviews to uncover insights through text analysis, sentiment analysis, and frequency distribution of words. Using Natural Language Processing (NLP) techniques and tools such as NLTK and scikit-learn, I aimed to process and analyze the data to understand common themes, sentiments, and the significance of certain words within the reviews. The goal was to create a system that could provide a deeper understanding of the textual data, helping in sentiment analysis and identifying key trends in movie reviews.

## Data Collection

The movie reviews were sourced using the Cinemagoer API, a Python library that interfaces with the Internet Movie Database (IMDb). This allowed for the retrieval of a wide array of reviews for specific movies, which were then stored for analysis. For instance, the movie "The Dark Knight" was selected for this project due to its popularity and the rich set of reviews it has garnered over the years.

## Implementation

At a high level, the project was implemented through a pipeline consisting of several major components:

- **Data Preparation and Cleaning:** Text data was read from a file and cleaned to remove punctuation and stopwords. This step was crucial for accurate frequency analysis and sentiment analysis later on.
- **Tokenization and Frequency Analysis:** The cleaned text was tokenized into words and analyzed for frequency distribution, identifying the most commonly used words in the reviews.
- **TF-IDF Calculation:** I calculated the Term Frequency-Inverse Document Frequency (TF-IDF) scores to understand the importance of words across documents.
- **Sentiment Analysis:** Utilizing NLTK's Sentiment Intensity Analyzer, I performed sentiment analysis on each review, providing a compound score to gauge the overall sentiment.

A significant design decision involved choosing between a simple frequency analysis and integrating TF-IDF for word importance. I opted to include both, as frequency analysis highlights common words, while TF-IDF offers insight into each review's unique words.

Throughout the project, ChatGPT was instrumental in optimizing code and clarifying concepts, especially when implementing the sentiment analysis component and fine-tuning the TF-IDF calculations.

## Results

The project successfully analyzed the dataset to reveal:

- The top 10 most frequently occurring words provided insight into common themes across movie reviews.
- The top 10 words by TF-IDF score highlight terms uniquely significant in specific reviews but not common across the dataset.
- Sentiment analysis across reviews, visualizing sentiment scores indicating the overall sentiment trend among the reviews.

Visualizations played a key role in presenting my findings, making the insights accessible and understandable. Figures included bar charts for word frequencies, TF-IDF scores, and sentiment scores, each offering a different perspective on the data.

## Reflection

The project was a valuable learning experience, especially in applying NLP techniques to real-world data. While challenging, cleaning and preparing the data was crucial for the accuracy of my analysis. Implementing TF-IDF and sentiment analysis provided deeper insights than a straightforward frequency analysis could have offered.

One area for improvement is the scalability of my implementation. As the dataset grows, optimizing performance and managing resources will become increasingly important.

From a learning perspective, this project reinforced the importance of thorough data preparation and the power of NLP in extracting meaningful insights from text. ChatGPT was an invaluable resource for overcoming technical hurdles and enhancing my understanding of complex concepts. Moving forward, the knowledge gained from this project will be instrumental in tackling more advanced NLP tasks.

### Code Availability

The code used in this project has been optimized with the help of GPT-4 and Claude 3 Sonnet, demonstrating the practical application of advanced AI models in software development and data analysis.

```python

#=========
# Part I
#=========
from imdb import Cinemagoer

# create an instance of the Cinemagoer class
ia = Cinemagoer()

# search for a movie
movie = ia.search_movie("The Dark Knight")[0]
print(movie.movieID)
# Output: '0468569'

# Get reviews with additional info argument for fetching reviews
movie = ia.get_movie('0468569', info=['reviews'])
reviews = movie.get('reviews', [])

# Initialize an empty list to store formatted reviews
formatted_reviews = []

# Loop through the reviews with enumeration to number and store them
for i, review in enumerate(reviews, start=1):
    formatted_review = f"Review #{i}:\n{review['content']}\n"
    formatted_reviews.append(formatted_review)

# Now `formatted_reviews` contains all the reviews formatted. You can print them or save them to a file.
# For example, to print all saved reviews:
for review in formatted_reviews:
    print(review)

# To save the reviews to a file:
with open("movie_reviews.txt", "w", encoding="utf-8") as file:
    for review in formatted_reviews:
        file.write(review)


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
