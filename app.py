# Import necessary libraries
import pandas as pd
import re
import string
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob

# Install necessary packages
# !pip install nltk  # Uncomment if nltk is not installed
# !pip install Sastrawi  # Uncomment if Sastrawi is not installed
# !pip install transformers  # Uncomment if transformers is not installed
# !pip install wordcloud  # Uncomment if wordcloud is not installed


# Convert DataFrame to CSV
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Check the CSV file
def print_csv(file_path):
    data = pd.read_csv(file_path)
    print(data.head())

# Preprocessing text
def preprocess_text(kalimat):
    lower_case = kalimat.lower()
    hasil = re.sub(r"\d+", "", lower_case)
    hasil = hasil.translate(str.maketrans("","",string.punctuation))
    hasil = hasil.strip()
    hasil = re.sub('@[^\s]+', ' ', hasil)
    hasil = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", hasil)
    hasil = re.sub('<.*?>', ' ', hasil)
    hasil = re.sub('[^a-zA-Z0-9]', ' ', hasil)
    hasil = re.sub("\n", " ", hasil)
    hasil = re.sub(r"\b[a-zA-Z]\b", " ", hasil)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)
    hasil = emoji_pattern.sub(r'', hasil)
    hasil = ' '.join(hasil.split())
    return hasil

# Normalize text
def normalisasi_teks(teks, kamus):
    kata_asli = teks.split()
    kata_normal = []
    for kata in kata_asli:
        normal = kamus[kamus['kata_asli'] == kata]['kata_normal'].values
        if normal:
            kata_normal.append(normal[0])
        else:
            kata_normal.append(kata)
    return ' '.join(kata_normal)

# Tokenize text
def tokenize_text(kalimat):
    tokens = nltk.tokenize.word_tokenize(kalimat)
    return tokens

# Normalize terms
def normalized_term(document, normalized_word_dict):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]

# Stopword removal
def stopword_text(tokens, stopwords):
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens

# Stemming text
def stemming_text(tokens, stemmer):
    hasil = [stemmer.stem(token) for token in tokens]
    return hasil

# Perform sentiment analysis
def label_teks(teks, sentiment_analysis):
    results = sentiment_analysis(teks)
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    labels = [label_index[result['label']] for result in results]
    return labels

# Visualization
def visualize_sentiments(text):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (6, 6))
    sizes = [count for count in text['label'].value_counts()]
    labels = list(text['label'].value_counts().index)
    explode = (0.1, 0, 0)
    ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
    ax.set_title('Sentiment Polarity pada Data', fontsize = 16, pad = 20)
    plt.show()

def generate_wordcloud(text):
    from wordcloud import WordCloud
    list_words=''
    for tweet in text['teks']:
        for word in tweet:
            list_words += ''+(word)
    wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(list_words)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.set_title('Word Cloud of text Data', fontsize = 18)
    ax.grid(False)
    ax.imshow((wordcloud))
    fig.tight_layout(pad=0)
    ax.axis('off')
    plt.show()

# Main function
if __name__ == "__main__":
    import os

    # Load NLTK resources
    nltk.download('punkt')

    # Paths
    csv_file = 'detikcom_20240601_001841.csv'
    cleaned_csv_file = 'data_clean_detik.csv'
    labeled_csv_file = 'data_label_detik.csv'

    # Read CSV
    data = read_csv(csv_file)

    # Preprocess data
    if 'Date' in data.columns:
        data = data.drop('Date', axis=1)
    data = data.drop(['URL', 'Summary', 'Category', 'Day'], axis=1, errors='ignore')

    # Apply preprocessing
    data['text_clean'] = data['Title'].apply(preprocess_text)

    # Load normalization dictionary
    kamus = pd.read_csv('kamusalay.csv', header=None)
    kamus.columns = ['kata_asli', 'kata_normal']
    data['normal'] = data['text_clean'].apply(normalisasi_teks, kamus=kamus)

    # Tokenize text
    data['token'] = data['normal'].apply(tokenize_text)

    # Normalize terms
    normalized_word = pd.read_csv("Normalisasi.csv", encoding='latin1')
    normalized_word_dict = {}
    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]
    data['normal_2'] = data['token'].apply(normalized_term, normalized_word_dict=normalized_word_dict)

    # Stopword removal
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    data['stop'] = data['normal_2'].apply(stopword_text, stopwords=stopwords)

    # Stemming
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    data['stemmed'] = data['stop'].apply(stemming_text, stemmer=stemmer)

    # Save cleaned data
    data.to_csv(cleaned_csv_file, index=False)

    # Load for sentiment analysis
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Load cleaned data
    data = read_csv(cleaned_csv_file)

    # Perform sentiment analysis
    data['teks'] = data['stemmed'].astype(str)
    data['label'] = data['teks'].apply(lambda x: label_teks(x, sentiment_analysis))

    # Clean labels
    data['label'] = data['label'].astype(str)
    data['label'] = data['label'].str.replace('[', '').str.replace(']', '')
    data['label'] = data['label'].str.replace("'", "")

    # Save labeled data
    data.to_csv(labeled_csv_file, index=False)

    # Load labeled data
    text = read_csv(labeled_csv_file)

    # Visualize sentiments
    visualize_sentiments(text)

    # Generate word cloud
    generate_wordcloud(text)
