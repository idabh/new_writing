import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from nltk.tokenize import word_tokenize
import spacy

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

