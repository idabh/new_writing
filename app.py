import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.text import Text
import seaborn as sns
import numpy as np

from nltk.corpus import stopwords

import os
import re

from streamlit.components.v1 import html
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# managing NLTK data
nltk_data_dir = "./nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

# we add the spacy model in the same way
# import spacy

# spacy_dir = "./spacy_data"
# if not os.path.exists(spacy_dir):
#     os.makedirs(spacy_dir, exist_ok=True)
# # add the model in the dir
# cache_dir = os.path.join(spacy_dir, "en_core_web_sm")
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir, exist_ok=True)
# # download the model
# cache_dir=os.getenv("SPACY_DATA", "./spacy_data")
# model_path="en_core_web_sm"
# try:
#     nlp = spacy.load(os.path.join(cache_dir,model_path))
# except OSError:
#     spacy.cli.download(model_path)
#     nlp = spacy.load(model_path)
#     nlp.to_disk(os.path.join(cache_dir,model_path))

# Streamlit app
st.title("Text Analysis Workshop")

# Large text input field
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", height=300)

# Slider for adjustable parameters
st.sidebar.header("Adjustable Parameters")
window_size = st.sidebar.slider("Window Size for TTR Analysis", min_value=10, max_value=100, value=50, step=10)


# Load Lancaster Norms dataset and process it
url = 'https://raw.githubusercontent.com/seantrott/cs_norms/refs/heads/main/data/lexical/lancaster_norms.csv'
try:
    df = pd.read_csv(url)
    # st.write("Lancaster Norms Dataset loaded successfully!")
except Exception as e:
    st.error(f"Failed to load lancaster dataset: {e}")

lancaster_norms = df.set_index('Word').filter(like='.mean').to_dict(orient='index')
lancaster_norms = {word.lower(): {k.split('.')[0]: v for k, v in values.items()} for word, values in lancaster_norms.items()}

# Clean sensory dictionary
sensory_dict = {}
for word, values in lancaster_norms.items():
    sensory_dict[word] = {k: v for k, v in values.items() if k.capitalize() in ['Auditory', 'Olfactory', 'Gustatory', 'Interoceptive', 'Visual', 'Haptic']}

# Load Concreteness data
def load_concreteness_data_english():
    url = "https://raw.githubusercontent.com/josh-ashkinaze/Concreteness/refs/heads/master/concreteness_scores_original.csv"
    concreteness_df = pd.read_csv(url, sep=',', on_bad_lines='skip')
    concreteness_dict = pd.Series(concreteness_df['Conc.M'].values, index=concreteness_df['Word']).to_dict()
    return concreteness_dict

concreteness_dict = load_concreteness_data_english()

# Default text for the text area
default_text = """
This is a sample text. You can replace it with any content you'd like to analyze.
"""

# Helper function for clearing figures before plotting
def plot_sentence_lengths(sentences):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Length (words)')
    plt.title('Sentence Length Over Time')
    st.pyplot(plt)

# Plot TTR over time
def plot_ttr_over_time(tokens, window_size):
    ttr_values = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window_tokens = tokens[i:i + window_size]
        types = set(window_tokens)
        ttr = len(types) / len(window_tokens)
        ttr_values.append(ttr)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(ttr_values)), ttr_values, marker='o', linestyle='-', color='purple')
    plt.xlabel("Window Index")
    plt.ylabel("Type-Token Ratio (TTR)")
    plt.title("Type-Token Ratio Over Time")
    st.pyplot(plt)

# Plot word frequency distribution
def plot_word_frequency(most_common_words):
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Word Frequency Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Plot sentiment scores
def plot_sentiment(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if score > 0 else 'red' if score < 0 else 'yellow' for score in sentiment_scores])
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score per Sentence")
    st.pyplot(plt)

# Plot sentiment line
def plot_sentiment_line(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentiment_scores)), sentiment_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score Zigzag Line")
    st.pyplot(plt)

# Tokenize text if available
if user_text:
    try:
        tokens = word_tokenize(user_text.lower())
        sentences = sent_tokenize(user_text)
        nltk_text = Text(tokens)
    except LookupError:
        st.error("Required NLTK resources are missing. Please ensure 'punkt' is downloaded.")
        tokens, sentences = [], []
        nltk_text = None

    # Tabs for each analysis feature
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Word & Character Count", "Sentence Analysis", "Type-Token Ratio", "Word Frequency Distribution", 
        "Sentiment Analysis", "Part of Speech Tagging", "Concreteness Analysis", "Sensory Analysis", "Word2Vec Similar Words"])

    if tokens:
        with tab1:
            st.header("Word & Character Count")
            num_words = len(tokens)
            num_chars = len(user_text)
            avg_word_length = sum(len(word) for word in tokens) / num_words if num_words > 0 else 0
            st.write(f"Total Words: {num_words}")
            st.write(f"Total Characters: {num_chars}")
            st.write(f"Average Word Length: {avg_word_length:.2f} characters")
            longest_words = sorted(tokens, key=len, reverse=True)[:5]
            st.write("5 Longest Words:")
            for word in longest_words:
                st.write(f"{word} ({len(word)} characters)")

            # Concordance and Dispersion Plot
            if nltk_text:
                search_word = st.text_input("Enter a word to find its dispersion plot:")
                if search_word:
                    try:
                        st.write("Dispersion Plot for selected words:")
                        plt.figure(figsize=(10, 5))
                        nltk_text.dispersion_plot([search_word])
                        st.pyplot(plt)

                    except ValueError:
                        st.write("Word not found in the text.")

        with tab2:
            st.header("Sentence Analysis")
            num_sentences = len(sentences)
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            st.write(f"Total Sentences: {num_sentences}")
            st.write(f"Average Sentence Length: {avg_sentence_length:.2f} words")
            if num_sentences > 0:
                longest_sentences = sorted(sentences, key=len, reverse=True)[:2]
                st.write("Longest Sentences:")
                st.write(longest_sentences[0])
                if len(longest_sentences) > 1:
                    st.write(longest_sentences[1])
                plot_sentence_lengths(sentences)

        with tab3:
            st.header("Type-Token Ratio")
            types = set(tokens)
            ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
            st.write(f"Type-Token Ratio (TTR): {ttr:.2f}")
            if len(tokens) >= window_size:
                st.write("Type-Token Ratio Over Time (using a sliding window):")
                plot_ttr_over_time(tokens, window_size)

            hapax_legomena = [word for word in tokens if tokens.count(word) == 1]
            st.write(f"Number of Hapax Legomena (words that occur only once): {len(hapax_legomena)}")
            st.write(f"Percentage of Hapax Legomena: {(len(hapax_legomena) / len(tokens)) * 100:.2f}%")

        with tab4:
            st.header("Word Frequency Distribution")
            fdist = FreqDist(tokens)
            most_common_words = fdist.most_common(10)
            plot_word_frequency(most_common_words)

        with tab5:
            st.header("Sentiment Analysis")
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
            plot_sentiment(sentiment_scores)

        with tab6:
            st.header("Part of Speech Tagging")
            pos_tags = nltk.pos_tag(tokens, tagset="universal")
            st.write("First 10 tokens and their POS tags:")
            st.write(pos_tags[:10])

        with tab7:
            st.header("Concreteness Analysis")
            if user_text:
                st.write("Concreteness Scores (first 10 words):")
                concreteness_scores = [concreteness_dict.get(word, "No data") for word in tokens[:10]]
                for word, score in zip(tokens[:10], concreteness_scores):
                    st.write(f"{word}: {score}")

        with tab8:
            st.header("Sensory Analysis")
            if user_text:
                sensory_scores = {word: sensory_dict.get(word, {}) for word in tokens}
                st.write("Sensory Scores (first 10 words):")
                for word, score in list(sensory_scores.items())[:10]:
                    st.write(f"{word}: {score}")

        with tab9:
            st.header("Word2Vec Similar Words")
            st.write("Feature Coming Soon!")
