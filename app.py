import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
#nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.text import Text

# we add the spacy model in the same way
import spacy
nlp = spacy.load("en_core_web_sm", exclude=["ner"])


# just for setting plotting layout
sns.set_theme(style="whitegrid")

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
    plt.xlabel('Sentence Number')
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
    # rotate x-axis labels for better readability
    plt.xticks(rotation=45)
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

def plot_concreteness_per_sentence(sentences):
    concreteness_scores = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        concreteness_scores.append(sum(concreteness_dict.get(token, 0) for token in tokens) / len(tokens) if len(tokens) > 0 else 0)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(concreteness_scores)), concreteness_scores, marker='o', linestyle='-', color='orange')
    plt.xlabel("Sentence Number")
    plt.ylabel("Concreteness Score")
    plt.title("Concreteness Score per Sentence")
    st.pyplot(plt)

# a barplot for the sensory analysis
def plot_sensory_analysis(sensory_scores):
    senses = list(sensory_scores.keys())
    values = list(sensory_scores.values())
    # get a nice set of colors
    colors = sns.color_palette("husl", len(senses))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=senses, y=values, palette=colors)
    plt.title("Sensory Analysis")
    plt.xlabel("Sensory Type")
    plt.ylabel("Average Score")
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
            st.write(f"*Total Words:* {num_words}")
            st.write(f"*Total Characters:* {num_chars}")
            st.write(f"*Average Word Length:* {avg_word_length:.2f} characters")
            longest_words = sorted(tokens, key=len, reverse=True)[:5]
            st.write("*5 Longest Words:*")
            for word in longest_words:
                st.write(f"{word} ({len(word)} characters)")

            # Concordance and Dispersion Plot
            if nltk_text:
                search_word = st.text_input("Enter a word to find its dispersion plot:")
                st.write("to look up more words, separate them by commas")
                if search_word:
                    # see if it's a list seperated by commas, if it is, plot all words,
                    # if not, just use the word
                    try:
                        search_words = search_word.replace(" ", "").split(",")
                    # if not more than one word, just plot the one word
                    except AttributeError:
                        search_words = [search_word]

                    try:
                        # always include "and" and "the" as baseline words
                        baseline_words = ["and", "the"]
                        all_words = search_words + baseline_words
                        st.write("Dispersion Plot for selected words:")
                        plt.figure(figsize=(10, 5))
                        nltk_text.dispersion_plot(all_words, colors=colors)
                        st.pyplot(plt)
                    except ValueError:
                        st.write("Word not found in the text.")

            # if nltk_text:
            #     search_word = st.text_input("Enter a word to find its dispersion plot:")
            #     st.write("To look up more words, separate them by commas")
            #     if search_word:
            #         # Parse input words, ensuring "and" and "the" are included
            #         try:
            #             search_words = search_word.replace(" ", "").split(",")
            #         except AttributeError:
            #             search_words = [search_word]
                    
            #         # Always include "and" and "the" as baseline words
            #         baseline_words = ["and", "the"]
            #         all_words = baseline_words + search_words  # Ensure they're always plotted
                    
            #         try:
            #             st.write("Dispersion Plot for selected words (baseline: 'and', 'the'):")
            #             plt.figure(figsize=(10, 5))

            #             # Draw baseline words in orange
            #             baseline_color = 'orange'
            #             for baseline_word in baseline_words:
            #                 positions = [i for i, word in enumerate(nltk_text) if word == baseline_word]
            #                 plt.plot(positions, [baseline_word] * len(positions), '|', color=baseline_color, label=f"'{baseline_word}' (baseline)")

            #             # Draw user-provided words in default color
            #             for word in search_words:
            #                 positions = [i for i, word in enumerate(nltk_text) if word == word]
            #                 plt.plot(positions, [word] * len(positions), '|', label=word)

            #             plt.title("Dispersion Plot")
            #             plt.xlabel("Word Offset")
            #             plt.ylabel("Words")
            #             plt.legend(loc='upper right')
            #             st.pyplot(plt)
            #         except ValueError:
            #             st.write("Word not found in the text.")

                    # try:
                    #     st.write("Dispersion Plot for selected words:")
                    #     plt.figure(figsize=(10, 5))
                    #     nltk_text.dispersion_plot([search_word])
                    #     st.pyplot(plt)

                    # except ValueError:
                    #     st.write("Word not found in the text.")

        with tab2:
            st.header("Sentence Analysis")
            num_sentences = len(sentences)
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            st.write(f"Total Sentences: {num_sentences}")
            st.write(f"Average Sentence Length: {avg_sentence_length:.2f} words")
            if num_sentences > 0:
                longest_sentences = sorted(sentences, key=len, reverse=True)[:2]
                st.write("**Longest Sentence(s):**")
                st.write(f"*{longest_sentences[0]}*")
                if len(longest_sentences) > 1:
                    st.write(f"*{longest_sentences[1]}*")
                # add shortest sentence
                shortest_sentences = sorted(sentences, key=len)[:2]
                # make it bold
                st.write("**Shortest Sentence(s):**")
                st.write(f"*{shortest_sentences[0]}*")
                if len(shortest_sentences) > 1:
                    st.write(f"*{shortest_sentences[1]}*")
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
            st.write(f"Percentage of Hapax Legomena: {(len(hapax_legomena) / len(tokens)) * 100:.1f}%")

        with tab4:
            st.header("Word Frequency Distribution")
            fdist = FreqDist(tokens)
            most_common_words = fdist.most_common(20)
            plot_word_frequency(most_common_words)

        with tab5:
            st.header("Sentiment Analysis")
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
            plot_sentiment(sentiment_scores)

        with tab6:
            st.header("Part of Speech Tagging")
            # use spacy to tag the parts of speech
            doc = nlp(user_text)
            pos_tags = [(token.text, token.pos_) for token in doc]
            #pos_tags = nltk.pos_tag(tokens, tagset="universal")
            st.write("First 10 tokens and their POS tags:")
            st.write(pos_tags[:10])

        with tab7:
            st.header("Concreteness Analysis")
            if user_text:
                st.write("Concreteness Scores (first 10 words):")
                concreteness_scores = [concreteness_dict.get(word, "No data") for word in tokens[:10]]
                for word, score in zip(tokens[:10], concreteness_scores):
                    st.write(f"{word}: {score}")
                # added concreteness per sentence
                plot_concreteness_per_sentence(sentences)

        with tab8:
            st.header("Sensory Analysis")
            if user_text:
                # sensory_scores = {word: sensory_dict.get(word, {}) for word in tokens}
                # st.write("Sensory Scores (first 10 words):")
                # for word, score in list(sensory_scores.items())[:10]:
                #     st.write(f"{word}: {score}")

                # lemmatize the tokens
                # use spacy to lemmatize
                doc = nlp(user_text)
                lemmatized_tokens = [token.lemma_ for token in doc]

                #lemma_types = list(set(lemmatized_tokens))
                # Collect sensory data for tokens in user text
                sensory_values = {'Auditory': [], 'Olfactory': [], 'Gustatory': [], 'Interoceptive': [], 'Visual': [], 'Haptic': []}
                for token in lemmatized_tokens:
                    if token in sensory_dict:
                        for sense, value in sensory_dict[token].items():
                            sensory_values[sense].append((value, token))
                # Calculate and display the average for each sense
                avg_sensory_values = {sense: sum([value for value, _ in values]) / len(values) if values else np.nan for sense, values in sensory_values.items()}
                st.write("Average Sensory Values:")
                plot_sensory_analysis(avg_sensory_values)

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
                    # just write the values and words as a list
                    st.write([f"{word}: {value:.2f}" for value, word in top_values])
                    # for value, word in top_values:
                    #     st.write(f"{word}: {value:.2f}")

                

        with tab9:
            st.header("Word2Vec Similar Words")
            st.write("Feature Coming Soon!")
            #st.header("Word2Vec Similar Words (Drafty)")
            # import_button = st.button("Import Word2Vec Model")
            # if 'model' not in st.session_state and import_button:
            #     from gensim.models import KeyedVectors
            #     import gensim.downloader as api
            #     st.write("Loading Word2Vec model... This may take a while.")
            #     st.session_state.model = api.load("word2vec-google-news-300")
            #     #api.load("glove-wiki-gigaword-50")
            #     # load word2vec from online
            #     st.write("Model loaded successfully!")
            #     # types
            #     filtered_tokens = [token for token in set(tokens) if token in model.key_to_index]
            # if 'model' in st.session_state:
            #     model = st.session_state.model
            #     if st.button("Create Clusters w Number"):
            #         num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            #         # Filter tokens to only include those in the Word2Vec vocabulary
            #         if filtered_tokens:
            #             # Get Word2Vec embeddings for the filtered tokens
            #             word_vectors = np.array([model[token] for token in filtered_tokens])
                        
            #             # Perform KMeans clustering
            #             kmeans = KMeans(n_clusters=num_clusters, random_state=2)
            #             kmeans.fit(word_vectors)
            #             labels = kmeans.labels_
            #             # Create clusters and display the most relevant words per cluster
            #             clusters = {i: [] for i in range(num_clusters)}
            #             for idx, label in enumerate(labels):
            #                 clusters[label].append(filtered_tokens[idx])
            #             st.write("Word Clusters:")
            #             for cluster_id, words in clusters.items():
            #                 st.write(f"Cluster {cluster_id + 1}: {', '.join(words)}")
            #         else:
            #             st.write("No suitable words found in the text for clustering.")
            # if st.button("AfProp"):
            #     if filtered_tokens:
            #         # import affinity propagation
            #         from sklearn.cluster import AffinityPropagation
            #         # Get Word2Vec embeddings for the filtered tokens
            #         embeddings = np.array([model[token] for token in filtered_tokens])
            #         similarity_matrix = cosine_similarity(embeddings)
            #         # Perform Affinity Propagation clustering
            #         affprop = AffinityPropagation(affinity='precomputed', damping=0.9, random_state=42)
            #         affprop.fit(similarity_matrix)
            #         # Print clusters
            #         clusters = {}
            #         for word, cluster_id in zip(filtered_tokens, affprop.labels_):
            #             clusters.setdefault(cluster_id, []).append(word)
            #         # Display number of clusters
            #         st.write(f"Number of clusters: {len(clusters)}")
            #         # Display clusters
            #         st.write("Word Clusters:")
            #         for cluster_id, words in clusters.items():
            #             st.write(f"Cluster {cluster_id + 1}: {', '.join(words)}")
