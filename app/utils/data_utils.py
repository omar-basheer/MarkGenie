import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    """
    Preprocesses text by removing punctuation, converting to lowercase, and removing stop words.

    Args:
    - text (str): Input text to be preprocessed.

    Returns:
    - preprocessed_text (str): Preprocessed text.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert to lowercase
    text = text.lower()

    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    preprocessed_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Reassemble the preprocessed tokens into text
    preprocessed_text = ' '.join(preprocessed_tokens)

    return preprocessed_text
