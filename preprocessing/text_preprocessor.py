import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob

# Function to download required NLTK resources only if not already present
def download_nltk_resources():
    resources = ['punkt','punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')  # Check if resource is already present
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

# Download NLTK resources
download_nltk_resources()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error during text preprocessing: {str(e)}")
            return ""

    def get_sentiment(self, text):
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return {'polarity': None, 'subjectivity': None}
    
    def extract_entities(self, text):
        try:
            blob = TextBlob(text)
            return {
                'noun_phrases': blob.noun_phrases,
                'tags': [tag for word, tag in blob.tags]
            }
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return {'noun_phrases': [], 'tags': []}

