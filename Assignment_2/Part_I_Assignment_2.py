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