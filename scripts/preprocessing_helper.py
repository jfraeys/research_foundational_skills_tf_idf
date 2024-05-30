import re
from multiprocessing import Pool

import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet

nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

ENGLISH_WORD_SET = set(words.words())

def clean_text(text):
    # Remove website URLs
    text = re.sub(r"https?:\/\/.*", "", text)

    # Remove special characters and digits
    text = re.sub(r"[\d\W]+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Replace periods not followed by a space with a space
    text = re.sub(r"\.(?!\s|$)", ". ", text)

    # Remove tokens between two or more consecutive underscores
    text = re.sub(r"(_{2,}[^_]*_{2,})", "", text)

    # Remove any extra whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip().lower()

    return text

def preprocess_text(text: pd.Series) -> pd.Series:
    processed_text = text.progress_apply(clean_text)
    processed_text = processed_text.progress_apply(
        tokenize_text
    )
    processed_text = processed_text.progress_apply(
        lemmatize, check_pos=True
    )

    return processed_text.str.join(" ")

def preprocess_labels(labels):
    preprocessed_labels = []
    for label in labels:
        tokens = label.split('-')  # Split hyphenated words
        lemmatized_tokens = lemmatize(tokens)
        preprocessed_label = ' '.join(lemmatized_tokens)
        preprocessed_labels.append(preprocessed_label)
    return preprocessed_labels

def get_wordnet_pos(tag):
    # Map the POS tags from the NLTK library to WordNet POS tags.
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default return is NOUN


def lemmatize(tokens: list[str], check_pos=True):
    lemmatized = []
    wordnet_lemma = WordNetLemmatizer()

    if check_pos:
        tagged_tokens = nltk.pos_tag(tokens)
        for term, pos in tagged_tokens:
            # Lemmatize with consideration of POS tags
            lemma = wordnet_lemma.lemmatize(term, pos=get_wordnet_pos(pos))
            lemmatized.append(lemma)
    else:
        for term in tokens:
            # Lemmatize without considering POS tags
            lemma = wordnet_lemma.lemmatize(term)
            lemmatized.append(lemma)

    return lemmatized

def tokenize_text(text: str, remove_stopwords=True) -> list[str]:
    # Tokenize input text using NLTK library
    tokens = nltk.word_tokenize(text)

    # Get stopwords set for faster membership tests
    stopword_set = set(stopwords.words('english')) if remove_stopwords else set()

    return [
        token for token in tokens
        if not remove_stopwords or token not in stopword_set
    ]


def filter_english_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token in ENGLISH_WORD_SET]

def replace_keywords(text, mapping):
    # Compile regular expression patterns for each word in the mapping
    compiled_patterns = {word: [re.compile(fr'\b{re.escape(rep)}\b', re.IGNORECASE) for rep in rep_list] for word, rep_list in mapping.items()}

    for word, patterns in compiled_patterns.items():
        for pattern in patterns:
            # Find all words matching the pattern in the text
            matches = pattern.findall(text)

            if matches:
                # Create a list of categories containing the matched word
                categories_with_match = [category for category, rep_list in mapping.items() if any(word_in_rep for word_in_rep in rep_list for match in matches if word_in_rep in match)]
                # Replace each matched word with a randomly selected replacement word from the categories
                for match in matches:
                    choice = select_keyword(categories_with_match, mapping) if len(categories_with_match) > 1 else categories_with_match[0]
                    text = re.sub(r'\b{}\b'.format(re.escape(match)), choice , text)

    return text

def select_keyword(categories_with_match, mapping):
    if len(categories_with_match) > 1:
        # Generate a random number from a uniform distribution
        random_index = np.random.randint(low=0, high=len(categories_with_match))
        
        # Choose a category based on the random index
        chosen_category = categories_with_match[random_index]
        # Choose a replacement word randomly from the chosen category
        return chosen_category

def tokenize_batch_helper(df_batch: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df_batch[text_column + "_processed"] = df_batch[text_column].apply(tokenize_text)
    return df_batch

def tokenize_batch(df: pd.DataFrame, num_cores: int = 4) -> pd.DataFrame:
    batch_size = len(df) // num_cores
    df_split = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

    pool = Pool(num_cores)
    result_batches = pool.map(tokenize_batch_helper, df_split)
    pool.close()
    pool.join()

    df_result = pd.concat(result_batches, ignore_index=True)
    return df_result