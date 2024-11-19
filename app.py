import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.text import Text
import seaborn as sns
import re
from streamlit.components.v1 import html
# import spacy
import numpy as np
from nltk.corpus import stopwords
import os


# nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk_data_dir = "./nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)

nltk_downloads = ['punkt', 'vader_lexicon', 'averaged_perceptron_tagger', 'universal_tagset', 'stopwords']

for download_thing in nltk_downloads:
    try:
        nltk.download(download_thing, download_dir=nltk_data_dir)
    except:
        pass

# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('stopwords')

# URL of the dataset
url = 'https://raw.githubusercontent.com/seantrott/cs_norms/refs/heads/main/data/lexical/lancaster_norms.csv'

# Load the dataset into a pandas DataFrame
try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Failed to load dataset: {e}")

# Turn it into a dictionary containing the word as key and the content of the columns ending in '.mean' as values
lancaster_norms = df.set_index('Word').filter(like='.mean').to_dict(orient='index')

# Clean up dictionary keys for better usability
lancaster_norms = {word.lower(): {k.split('.')[0]: v for k, v in values.items()} for word, values in lancaster_norms.items()}

# Only keep sensory columns we're interested in
sensory_dict = {}
for word, values in lancaster_norms.items():
    new_values = {k: v for k, v in values.items() if k.capitalize() in ['Auditory', 'Olfactory', 'Gustatory', 'Interoceptive', 'Visual', 'Haptic']}
    sensory_dict[word] = new_values

# Load concreteness data
def load_concreteness_data_english():
    url = "https://raw.githubusercontent.com/josh-ashkinaze/Concreteness/refs/heads/master/concreteness_scores_original.csv"
    concreteness_df = pd.read_csv(url, sep=',', on_bad_lines='skip')
    concreteness_dict = pd.Series(concreteness_df['Conc.M'].values, index=concreteness_df['Word']).to_dict()
    return concreteness_dict

concreteness_dict = load_concreteness_data_english()

# Default text for the text area
default_text = """

"""

# Clear figures before plotting to avoid overlapping
def plot_sentence_lengths(sentences):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Length (words)')
    plt.title('Sentence Length Over Time')
    st.pyplot(plt)


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


def plot_word_frequency(most_common_words):
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Word Frequency Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)


def plot_sentiment(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if score > 0 else 'red' if score < 0 else 'yellow' for score in sentiment_scores])
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score per Sentence")
    st.pyplot(plt)


def plot_sentiment_line(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentiment_scores)), sentiment_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score Zigzag Line")
    st.pyplot(plt)


# Streamlit app
st.title("Text Analysis Workshop")

# Large text input field
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", value=default_text, height=300)

# Slider for adjustable parameters
st.sidebar.header("Adjustable Parameters")
window_size = st.sidebar.slider("Window Size for TTR Analysis", min_value=10, max_value=100, value=50, step=10)

if user_text:
    # Tokenize text
    try:
        tokens = wordpunct_tokenize(user_text.lower())
        sentences = sent_tokenize(user_text)
        nltk_text = Text(tokens)
    except LookupError:
        st.error("Required NLTK resources are missing. Please ensure 'punkt' is downloaded.")
        tokens, sentences = [], []
        nltk_text = None

    # Create tabs for each analysis feature
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Word & Character Count", "Sentence Analysis", "Type-Token Ratio", "Word Frequency Distribution", "Sentiment Analysis", "Part of Speech Tagging", "Concreteness Analysis", "Sensory Analysis", "Word2Vec Similar Words"])

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

        # Plot TTR over time with adjustable window size
        if len(tokens) >= window_size:
            st.write("Type-Token Ratio Over Time (using a sliding window):")
            plot_ttr_over_time(tokens, window_size)

        # Display lexical richness metrics
        hapax_legomena = [word for word in tokens if tokens.count(word) == 1]
        st.write(f"Number of Hapax Legomena (words that occur only once): {len(hapax_legomena)}")
        st.write(f"Percentage of Hapax Legomena: {(len(hapax_legomena) / len(tokens)) * 100:.2f}%" if len(tokens) > 0 else "Percentage of Hapax Legomena: 0%")

    with tab4:
        st.header("Word Frequency Distribution")
        if tokens:
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            freq_dist = FreqDist(filtered_tokens)
            most_common_words = freq_dist.most_common(10)
            st.write("Top 10 Most Frequent Words (excluding stopwords):")
            for word, count in most_common_words:
                st.write(f"{word}: {count} occurrences")
            plot_word_frequency(most_common_words)

    with tab5:
        st.header("Sentiment Analysis")
        if sentences:
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if len(sentiment_scores) > 0 else 0
            std_sentiment = np.std(sentiment_scores) if len(sentiment_scores) > 0 else 0
            st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
            st.write(f"Standard Deviation of Sentiment Scores: {std_sentiment:.2f}")
            if avg_sentiment > 0:
                st.write("Overall Sentiment: Positive")
            elif avg_sentiment < 0:
                st.write("Overall Sentiment: Negative")
            else:
                st.write("Overall Sentiment: Neutral")
            plot_sentiment(sentiment_scores)
            plot_sentiment_line(sentiment_scores)

    with tab6:
        st.header("Part of Speech Tagging")
        # doc = nlp(user_text)
        # adjectives = 0
        # nouns = 0
        # verbs = 0

        # for token in doc:
        #     if token.pos_ == "ADJ":
        #         adjectives += 1
        #     elif token.pos_ == "NOUN":
        #         nouns += 1
        #     elif token.pos_ == "VERB":
        #         verbs += 1

        # nominal_ratio = (adjectives + nouns) / verbs if verbs > 0 else 0

        # st.write(f"Number of Adjectives: {adjectives}")
        # st.write(f"Number of Nouns: {nouns}")
        # st.write(f"Number of Verbs: {verbs}")
        # st.write(f"Nominal Ratio (Adjectives + Nouns) / Verbs: {nominal_ratio:.2f}")

        # # Buttons to remove adjectives, nouns, or verbs
        # if st.button("Remove All Adjectives"):
        #     filtered_text = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
        #     st.text_area("Text without Adjectives:", value=filtered_text, height=200)

        # if st.button("Remove All Nouns"):
        #     filtered_text = " ".join([token.text for token in doc if token.pos_ != "NOUN"])
        #     st.text_area("Text without Nouns:", value=filtered_text, height=200)

        # if st.button("Remove All Verbs"):
        #     filtered_text = " ".join([token.text for token in doc if token.pos_ != "VERB"])
        #     st.text_area("Text without Verbs:", value=filtered_text, height=200)

    with tab7:
        st.header("Concreteness Analysis")
        if tokens:
            # make the types first
            lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
            lemma_types = list(set(lemmatized_words))
            concreteness_scores = [concreteness_dict.get(word, None) for word in lemma_types if word in concreteness_dict]
            concreteness_scores = [score for score in concreteness_scores if score is not None]
            avg_concreteness = sum(concreteness_scores) / len(concreteness_scores) if concreteness_scores else 0
            st.write(f"Average Concreteness Score: {avg_concreteness:.2f}")

            sorted_tokens = sorted([(word, concreteness_dict[word]) for word in lemma_types if word in concreteness_dict], key=lambda x: x[1])
            most_abstract = sorted_tokens[:5]
            most_concrete = sorted_tokens[-5:]

            st.write("\n*\n5 Most Abstract Words (lemmatized):")
            for word, score in most_abstract:
                st.write(f"{word}: {score}")

            st.write("\n*\n5 Most Concrete Words (lemmatzied):")
            for word, score in most_concrete:
                st.write(f"{word}: {score}")

    with tab8:
            st.header("Sensory Analysis (Drafty)")

            # lemmatize the tokens
            lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            #lemma_types = list(set(lemmatized_tokens))

            # Collect sensory data for tokens in user text
            sensory_values = {'Auditory': [], 'Olfactory': [], 'Gustatory': [], 'Interoceptive': [], 'Visual': [], 'Haptic': []}
            for token in lemmatized_tokens:
                if token in sensory_dict:
                    for sense, value in sensory_dict[token].items():
                        sensory_values[sense].append((value, token))

            # Calculate and display the average for each sense
            avg_sensory_values = {sense: sum([value for value, _ in values]) / len(values) if values else 0 for sense, values in sensory_values.items()}
            st.write("Average Sensory Values:")
            for sense, avg_value in avg_sensory_values.items():
                st.write(f"{sense}: {avg_value:.2f}")

            senses_emoji = {
                "Visual": "👁️",
                "Auditory": "👂",
                "Haptic": "🤲",
                "Gustatory": "👅",
                "Olfactory": "👃",
                "Interoceptive": "🧠"
            }

            # for each sense, display the top 5 words avoiding duplicates
            st.write("\n**\nTop 5 Words per Sense:")
            for sense, values in sensory_values.items():
                unique_values = list(set(values))
                top_values = sorted(unique_values, key=lambda x: x[0], reverse=True)[:5]
                st.write("\n*\n")
                st.write(f"{senses_emoji[sense]} {sense}:")
                for value, word in top_values:
                    st.write(f"{word}: {value:.2f}")

    with tab9:
        st.write("hello")


