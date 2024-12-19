pip install nltk scikit-learn googletrans==4.0.0-rc1 emoji
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import movie_reviews
from googletrans import Translator
import emoji
import re

# Download NLTK resources
nltk.download('movie_reviews')

# Load dataset
def load_movie_review_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    data = pd.DataFrame(documents, columns=['review', 'sentiment'])
    data['review'] = data['review'].apply(lambda words: ' '.join(words))  # Combine words into full sentences
    return data

# Slang dictionary
slang_dict = {
    "brb": "be right back",
    "idk": "I don't know",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "smh": "shaking my head",
    "btw": "by the way",
    "imho": "in my humble opinion"
}

# Emoji sentiment mapping
emoji_dict = {
    "😀": "positive",
    "😄": "positive",
    "😊": "positive",
    "😍": "positive",
    "😢": "negative",
    "😡": "negative",
    "😠": "negative",
    "😒": "negative"
}

# Aspect keywords
aspect_keywords = {
    "acting": ["acting", "performance", "actor", "actress"],
    "direction": ["direction", "director", "cinematography"],
    "storyline": ["storyline", "plot", "script", "narrative"],
    "music": ["music", "soundtrack", "songs", "background score"],
    "visuals": ["visuals", "graphics", "effects", "animation"]
}

# Text preprocessing
def preprocess_text(text):
    # Translate emojis to sentiments
    text = emoji.demojize(text)
    for emoj, sentiment in emoji_dict.items():
        text = text.replace(emoj, sentiment)

    # Replace slang with full forms
    words = text.split()
    text = ' '.join([slang_dict[word.lower()] if word.lower() in slang_dict else word for word in words])

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load data
data = load_movie_review_data()
data['review'] = data['review'].apply(preprocess_text)
X = data['review']
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Multilingual translation and prediction
translator = Translator()

def translate_to_english(text):
    try:
        translated_text = translator.translate(text, src='auto', dest='en').text
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Aspect-based sentiment analysis
def analyze_aspects(text):
    aspect_sentiments = {}
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text.lower():
                aspect_sentiments[aspect] = predict_sentiment(text)
                break
    return aspect_sentiments

def predict_sentiment(input_text):
    # Translate to English if not already
    translated_text = translate_to_english(input_text)
    # Preprocess text
    processed_text = preprocess_text(translated_text)
    # Predict sentiment
    input_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(input_vectorized)[0]
    return prediction

# Test the system
if __name__ == "__main__":
    print("Sentiment Analysis Model with Emoji, Slang, Multilingual, and Aspect-Based Support (type 'exit' to quit)")

    while True:
        user_input = input("\nEnter a review for sentiment analysis: ")

        if user_input.lower() == 'exit':
            print("Exiting the model.")
            break

        sentiment = predict_sentiment(user_input)
        aspects = analyze_aspects(user_input)

        print(f"\nOverall Sentiment: {sentiment}")
        print("Aspect-Based Sentiments:")
        for aspect, aspect_sentiment in aspects.items():
            print(f" - {aspect.capitalize()}: {aspect_sentiment}")
