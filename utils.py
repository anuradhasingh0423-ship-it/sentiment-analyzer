import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()



def remove_emojis(text):
    emoji_pattern = re.compile("["               
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)




def handle_negations(text):
    words = text.split()
    new_words = []
    negation = False
    negation_scope = 0

    for w in words:
        if w in ["not", "no", "never"]:
            negation = True
            negation_scope = 3  # apply negation to next 3 words
            new_words.append("not")
            continue

        if negation and negation_scope > 0:
            new_words.append("not_" + w)
            negation_scope -= 1

            # End negation if scope is over
            if negation_scope == 0:
                negation = False

        else:
            new_words.append(w)

    return " ".join(new_words)



def clean_text(text):
    # lowercase
    text = text.lower()

    # Remove emojis
    text = remove_emojis(text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Apply negation handling
    text = handle_negations(text)

    # Token-level processing
    tokens = text.split()

    # Apply stemming
    tokens = [stemmer.stem(t) for t in tokens]

    # Keep negation tokens even if they contain stopwords
    tokens = [
        t for t in tokens
        if (t not in stop_words) or t.startswith("not_")
    ]

    return " ".join(tokens)
