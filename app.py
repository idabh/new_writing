import streamlit as st
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

import os

from streamlit.components.v1 import html
import nltk

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# managing NLTK data
nltk_data_dir = "./nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir)

from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.text import Text

# we add the spacy model in the same way
import spacy
nlp = spacy.load("en_core_web_sm", exclude=["ner"])


# just for setting plotting layout
sns.set_theme(style="whitegrid")

# Streamlit app
st.title("Text Analysis Workshop")

# Default text for the text area
default_text = """
This is a sample text. You can replace it with any content you'd like to analyze.
"""

# Large text input field with a button
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", height=300)
# add a button to load the user text, make it orange
if st.button("Compute metrics", icon="🔍"):
    user_text = user_text

# make it possible to choose the hemingway text to insert
if st.button("Or: show Hemingway excerpt"):
    with open("hemingway.txt", "r") as file:
        hemingway_text = file.read()
        #user_text = hemingway_text
        # also set the text area to the hemingway text
        st.text_area("Beginning of *The Old Man and the Sea* – **copy and paste above**", value=hemingway_text, height=300)
        #st.code(hemingway_text, language="python") # this will give a copyable code block instead...
# or plath
if st.button("Or: show Plath excerpt"):
    with open("plath.txt", "r") as file:
        plath_text = file.read()
        #user_text = plath_text
        st.text_area("Chapter 1 of *The Bell Jar* – **copy and paste above**", value=plath_text, height=300)
        #st.code(plath_text, language="python") # this will give a copyable code block instead...

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

# Helper function for clearing figures before plotting
def plot_sentence_lengths(sentences):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
    plt.xlabel('Sentence Number')
    plt.ylabel('Sentence Length (words)')
    plt.title('Sentence Length Over Time')
    st.pyplot(plt)

# plot it as a plotly plot with hoverdata as the sentences (wrapped)
def plot_sentence_lengths_plotly(sentences):
    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]
    wrapped_sentences = []
    for sentence in sentences:
        wrapped_sentence = "-<br>".join([sentence[i:i+70] for i in range(0, len(sentence), 70)])
        wrapped_sentences.append(wrapped_sentence)

    # Create plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(sentences))),
        y=[len(sentence.split()) for sentence in sentences],
        mode='lines+markers',
        name='Sentence Length',
        hovertext=wrapped_sentences,
        hoverinfo='text'
    ))
    fig.update_layout(
        title="Sentence Length Over Time",
        xaxis_title="Sentence Number",
        yaxis_title="Sentence Length (words)"
    )
    st.plotly_chart(fig)


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
    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.title("Word Frequency Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    # rotate x-axis labels for better readability
    plt.xticks(rotation=90)
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

# can we make this a plotly plot where we can hover to see each sentence?
def plot_sentiment_plotly(sentiment_scores, sentences):
    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]
    wrapped_sentences = []
    for sentence in sentences:
        wrapped_sentence = "-<br>".join([sentence[i:i+70] for i in range(0, len(sentence), 70)])
        wrapped_sentences.append(wrapped_sentence)

    # Create plotly plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(sentiment_scores))),
        y=sentiment_scores,
        marker_color=['green' if score > 0 else 'red' if score < 0 else 'yellow' for score in sentiment_scores],
        hovertext=wrapped_sentences,
        hoverinfo='text'
    ))
    # add a horizontal line at 0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=len(sentiment_scores),
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="dashdot"
        )
    )
    fig.update_layout(
        xaxis_title="Sentence Number",
        yaxis_title="Sentiment"
    )
    st.plotly_chart(fig)

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

# can we make the concreteness a plotly plot where we can hover to see each sentence?
def plot_concreteness_per_sentence_plotly(sentences):
    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]
    concreteness_scores = []
    wrapped_sentences = []
    for sentence in sentences:
        lemmas = [token.lemma_ for token in nlp(sentence)]
        # Calculate score
        if lemmas:
            score = sum(concreteness_dict.get(lemma, 0) for lemma in lemmas) / len(lemmas)
        else:
            score = np.nan

        concreteness_scores.append(round(score,1))
        # Add line breaks to long sentences
        wrapped_sentence = "-<br>".join([sentence[i:i+70] for i in range(0, len(sentence), 70)])
        wrapped_sentences.append(wrapped_sentence)

    # create plotly plot and make hoverdata the sentence
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(concreteness_scores))),
        y=concreteness_scores,
        mode='lines+markers',
        name='Concreteness Score',
        hovertext=wrapped_sentences,
        hoverinfo='text'
    ))
    fig.update_layout(
        title="Concreteness Score per Sentence",
        xaxis_title="Sentence Number",
        yaxis_title="Concreteness Score"
    )
    st.plotly_chart(fig)

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

# piechaert for sentiment dist
def plot_sentiment_pie(sentiment_scores):
    # Count the number of positive, negative, and neutral sentences
    positive = sum([1 for score in sentiment_scores if score > 0])
    negative = sum([1 for score in sentiment_scores if score < 0])
    neutral = len(sentiment_scores) - positive - negative

    # Define data for the pie chart
    values = [positive, negative, neutral]
    labels = ["Positive", "Negative", "Neutral"]

    # Create a ring-shaped pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # Creates the "donut" effect
        textinfo='percent+label',  # Display percentages and labels
        marker=dict(colors=["#2ca02c", "#d62728", "#1f77b4"])  # Custom colors
    )])

    # Add title and adjust layout
    fig.update_layout(title="Sentiment Analysis Distribution", title_x=0.5)
    st.plotly_chart(fig)

# scatterplot w plotly for hapax legomena where hoverdata is word
def plot_hapax_legomena_scatterplot(hapax_legomena):
    # Calculate word lengths for the y-axis
    hapax_lengths = [len(word) for word in hapax_legomena]
    # Create plotly scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(hapax_legomena))),
        y=hapax_lengths,
        mode='markers',
        marker=dict(color='red'),
        text=hapax_legomena,
        hoverinfo='text',
        name='Hapax Legomena'
    ))
    fig.update_layout(
        title="Hapax Legomena Scatterplot (Word Length on Y-Axis)",
        xaxis_title="Word Index",
        yaxis_title="Word Length"
    )
    st.plotly_chart(fig)



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
        "Sentiment Analysis", "Part of Speech Tagging", "Concreteness Analysis", "Sensory Analysis", "📝 Writing experiment"])

    if tokens:
        with tab1:
            st.header("Word & Character Count")
            num_words = len(tokens)
            num_chars = len(user_text)
            avg_word_length = sum(len(word) for word in tokens) / num_words if num_words > 0 else 0
            st.write(f"**Total Words:** {num_words}")
            st.write(f"**Total Characters:** {num_chars}")
            st.write(f"**Average Word Length:** {avg_word_length:.2f} characters")
            longest_words = sorted(set(tokens), key=len, reverse=True)[:5]
            st.write("**5 Longest Words:**")
            for word in longest_words:
                st.write(f"{word} ({len(word)} characters)")

            st.write("\n#### 🔍 **Search for a word in the text to see its dispersion plot**")
            # Concordance and Dispersion Plot
            if nltk_text:
                search_word = st.text_input("Enter a word to find its dispersion plot   \n*Psst: to look up more words, separate them by commas [like so: 'lama, lemon, lion']*    \n*– we always include the frequent 'and' and 'the' as baseline words*")
                if search_word:
                    # remove leading and trailing spaces
                    search_word = search_word.strip()
                    # remove trailing commas or punctuation
                    search_word = search_word.rstrip(".")
                    search_word = search_word.rstrip(",")
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
                        # remove the words that are in the text
                        search_words = [word.lower() for word in search_words if word.lower() not in baseline_words]
                        all_words = list(set(search_words + baseline_words))
                        plt.figure(figsize=(10, 5))
                        nltk_text.dispersion_plot(all_words)
                        st.pyplot(plt)
                    except ValueError:
                        st.write("Sorry, this word(s) was not found in the text.")

        with tab2:
            st.header("Sentence Analysis")
            num_sentences = len(sentences)
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            st.write(f"**Total Sentences:** {num_sentences}")
            st.write(f"**Average Sentence Length:** {avg_sentence_length:.2f} words")
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
                #plot_sentence_lengths(sentences)
                plot_sentence_lengths_plotly(sentences)

        with tab3:
            st.header("Type-Token Ratio")
            types = set(tokens)
            ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
            st.write(f"**Type-Token Ratio (TTR):** {ttr:.2f}")
            if len(tokens) >= window_size:
                st.write("Type-Token Ratio Over Time (using a sliding window):")
                plot_ttr_over_time(tokens, window_size)

            hapax_legomena = [word for word in tokens if tokens.count(word) == 1]
            st.write(f"**Number of *Hapax Legomena* (words that occur only once):** {len(hapax_legomena)}")
            st.write(f"**Percentage of *Hapax Legomena*:** {(len(hapax_legomena) / len(tokens)) * 100:.1f}%")
            # show a list of the hapax legomena

            hapax_indices = [i for i, word in enumerate(tokens) if tokens.count(word) == 1]

            if st.button("Show Hapax Legomena in text"):
                if len(hapax_legomena) > 1:
                    plot_hapax_legomena_scatterplot(hapax_legomena)
                else:
                    st.write("Only one hapax legomena found in the text.")


        with tab4:
            st.header("Word Frequency Distribution")
            fdist = FreqDist(tokens)
            # make number of words to show a slider
            freq_no_words = st.slider("Number of Words to Show", min_value=5, max_value=50, value=10)
            most_common_words = fdist.most_common(freq_no_words)
            plot_word_frequency(most_common_words)


        with tab5:
            st.header("Sentiment Analysis")
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]

            # add the mean and std
            mean_sentiment = np.mean(sentiment_scores)
            std_sentiment = np.std(sentiment_scores)
            st.write(f"📊 **Mean Sentiment Score:** {mean_sentiment:.2f}")
            st.write(f"\n📈 **Standard Deviation:** {std_sentiment:.2f}")

            plot_sentiment_plotly(sentiment_scores, sentences)
            st.write("How much of the text is positive, negative, or neutral?")
            plot_sentiment_pie(sentiment_scores)


        with tab6:
            st.header("Part of Speech Tagging")
            doc = nlp(user_text)
            adjectives = 0
            nouns = 0
            verbs = 0
            for token in doc:
                if token.pos_ == "ADJ":
                    adjectives += 1
                elif token.pos_ == "NOUN":
                    nouns += 1
                elif token.pos_ == "VERB":
                    verbs += 1
            nominal_ratio = (adjectives + nouns) / verbs if verbs > 0 else 0
            st.write(f"✨ **Number of Adjectives:** {adjectives}")
            st.write(f"🪴 **Number of Nouns:** {nouns}")
            st.write(f"🏃🏾 **Number of Verbs:** {verbs}")
            st.write(f"⚖️ **Nominal Ratio (Adjectives + Nouns) / Verbs:** {nominal_ratio:.2f}")
            # Buttons to remove adjectives, nouns, or verbs
            if st.button("Show text without Adjectives"):
                filtered_text = " ".join([token.text for token in doc if token.pos_ != "ADJ"]).strip()
                st.text_area("Show text without Adjectives:", value=filtered_text, height=200)
            if st.button("Show text without Nouns"):
                filtered_text = " ".join([token.text for token in doc if token.pos_ != "NOUN"]).strip()
                st.text_area("Text without Nouns:", value=filtered_text, height=200)
            if st.button("Show text without Verbs"):
                filtered_text = " ".join([token.text for token in doc if token.pos_ != "VERB"]).strip()
                st.text_area("Text without Verbs:", value=filtered_text, height=200)

            # st.header("Part of Speech Tagging")
            # # use spacy to tag the parts of speech
            # doc = nlp(user_text)
            # pos_tags = [(token.text, token.pos_) for token in doc]
            # #pos_tags = nltk.pos_tag(tokens, tagset="universal")
            # st.write("First 10 tokens and their POS tags:")
            # st.write(pos_tags[:10])



        with tab7:
            st.header("Concreteness Analysis")
            if user_text:
                lemmatized_words = [token.lemma_ for token in doc]
                lemma_types = list(lemmatized_words)

                concreteness_scores = [concreteness_dict.get(word, None) for word in lemma_types if word in concreteness_dict]
                concreteness_scores = [score for score in concreteness_scores if score is not None]
                avg_concreteness = sum(concreteness_scores) / len(concreteness_scores) if concreteness_scores else 0

                st.write(f"🪨 **Average Concreteness Score:** {avg_concreteness:.2f}")

                lemma_types_set = set(list(lemma_types))

                sorted_tokens = sorted([(word, concreteness_dict[word]) for word in lemma_types_set if word in concreteness_dict], key=lambda x: x[1])
                # make the number of words to show a slider
                no_words = st.slider("*Number of Words to Show*", min_value=5, max_value=20, value=5)
                most_abstract = sorted_tokens[:no_words]
                most_concrete = sorted_tokens[-no_words:]
                # put two barplots side by side showing the five most abstract and five most concrete words
                # two subplots
                plt.figure(figsize=(10, 3))
                plt.subplot(1, 2, 1)
                sns.barplot(x=[word for word, _ in most_abstract], y=[score for _, score in most_abstract], palette="PuRd")
                plt.title("Most Abstract Words")
                plt.ylabel("Concreteness")
                # set ylim to 0,5
                plt.ylim(0, 5)
                plt.xticks(rotation=90)

                plt.subplot(1, 2, 2)
                sns.barplot(x=[word for word, _ in most_concrete], y=[score for _, score in most_concrete], palette="GnBu")
                plt.title("Most Concrete Words")
                plt.ylabel("Concreteness")
                plt.xticks(rotation=90)
                st.pyplot(plt)

                # st.write("\n*\n5 Most Abstract Words:")
                # for word, score in most_abstract:
                #     st.write(f"{word}: {score}")
                # st.write("\n*\n5 Most Concrete Words:")
                # for word, score in most_concrete:
                #     st.write(f"{word}: {score}")
                # added concreteness per sentence
                #plot_concreteness_per_sentence(sentences)
                plot_concreteness_per_sentence_plotly(sentences)

        with tab8:
            st.header("Sensory Analysis")
            if user_text:
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
                st.write("**Average Sensory Values:**")
                plot_sensory_analysis(avg_sensory_values)

                # get the average sensory values per sentence, for each sense
                sensory_values_per_sentence = {sense: [] for sense in sensory_values}
                for sentence in sentences:
                    tokens = word_tokenize(sentence.lower())
                    lemmatized_tokens = [token.lemma_ for token in nlp(sentence)]
                    for token in lemmatized_tokens:
                        if token in sensory_dict:
                            for sense, value in sensory_dict[token].items():
                                sensory_values_per_sentence[sense].append(value)
                # get the average sensory values per sentence
                avg_sensory_values = {sense: sum(values) / len(values) if values else np.nan for sense, values in sensory_values_per_sentence.items()}


                senses_emoji = {
                    "Visual": "👁️",
                    "Auditory": "👂",
                    "Haptic": "🤲",
                    "Gustatory": "👅",
                    "Olfactory": "👃",
                    "Interoceptive": "🧠"
                }
                # for each sense, display the top 5 words avoiding duplicates
                st.write("\n**Top 5 Words per Sense:**")
                for sense, values in sensory_values.items():
                    unique_values = list(set(values))
                    top_values = sorted(unique_values, key=lambda x: x[0], reverse=True)[:5]
                    #st.write("\n*\n")
                    st.write(f"{senses_emoji[sense]} {sense}:")
                    # just write the values and words as a list
                    st.write([f"{word}: {value:.2f}" for value, word in top_values])
                    # for value, word in top_values:
                    #     st.write(f"{word}: {value:.2f}")

                # plot the sensory values as a lineplot
                #plot_sensory_line(avg_sensory_values, sentences)


                

        with tab9:
            # here we make a new tab for a writing experiment, we want them to be able to extract all nouns and all sentiment words
            st.header("Writing Experiment")
            st.write("In this tab, you can experiment with words in the text you have in the textbox above. Extract all nouns or sentiment words below.")
            # make a button for extracting all nouns
            if st.button("Extract Nouns", icon="🪴"):
                # use spacy to tag the parts of speech
                doc = nlp(user_text)
                nouns = [token.text for token in doc if token.pos_ == "NOUN"]
                # join list to string
                nouns = ", ".join(nouns)
                st.write("Here are all the nouns in your text:")
                st.write(nouns)
            # make a button for extracting all sentiment words
            if st.button("Extract Sentiment Words", icon="❤️"):
                sia = SentimentIntensityAnalyzer()
                sentiment_words = [token.text for token in doc if sia.polarity_scores(token.text)['compound'] != 0]
                # order them by sentiment
                sentiment_words = sorted(sentiment_words, key=lambda x: sia.polarity_scores(x)['compound'])
                # join list to string
                sentiment_words = ", ".join(sentiment_words)
                st.write("Here are all the sentiment words in your text.   \n The words are ordered from the most negative to the most positive:")
                st.write(sentiment_words)

            # make a new text area for the user to experiment with
            st.write("Now you can experiment with your text. Try to write a new text and analyze it.")
            new_text = st.text_area("Write your experiment below:", height=300)
            # make it downloadable
            if st.button("Download Text"):
                st.download_button(label="", icon="💾", data=new_text, file_name="new_text.txt", mime="text/plain")





            # st.header("Word2Vec Similar Words")
            # st.write("Feature Coming Soon!")
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