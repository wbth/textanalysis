This Python script processes and analyzes textual data, primarily focusing on tasks such as text cleaning, normalization, tokenization, stopword removal, stemming, sentiment analysis, and visualization.

1. Library Imports
The script imports essential libraries, such as pandas for data manipulation, nltk for natural language processing, re for regular expressions, and Sastrawi for stemming and stopword removal in the Indonesian language.
TextBlob is imported for additional NLP tasks, and transformers are used for advanced sentiment analysis.
2. Reading CSV Data
Functions are defined as reading and printing data from a CSV file using Pandas.
3. Preprocessing
The preprocess_text function performs several cleaning tasks:
Converts text to lowercase.
Removes numbers, punctuation, whitespace, emojis, and special characters.
Strips unnecessary spaces and removes HTML tags and URLs.
4. Normalization
normalisasi_teks uses a dictionary to map and replace non-standard words or slang with their normalized forms.
normalized_term applies a similar normalization process at the token level.
5. Tokenization
The tokenize_text function splits text into individual words or tokens using nltk.
6. Stopword Removal
Stopwords (common words like "and," "is") are filtered out using Sastrawi.
7. Stemming
Words are reduced to their base form (e.g., "running" to "run") using the Sastrawi stemmer.
8. Sentiment Analysis
Sentiment analysis is performed using a pre-trained Indonesian sentiment analysis model (mdhugol/indonesia-bert-sentiment-classification) from Hugging Face's Transformers library.
Sentiments are categorized into positive, neutral, and negative using labels from the model output.
9. Visualization
Two visualization methods are defined:
Sentiment Distribution Pie Chart:
Displays the distribution of sentiment labels (positive, neutral, negative) in the dataset.
Word Cloud:
Creates a word cloud from the text data to visualize frequently occurring words.
10. Data Pipeline
The script processes data as follows:
Reads the CSV file containing raw text data.
Cleans and preprocesses the text, including normalization, tokenization, stopword removal, and stemming.
Saves the cleaned data to a new CSV file.
Performs sentiment analysis and labels the text.
Saves the labeled data to another CSV file.
Visualizes sentiment distribution and generates a word cloud.
11. Main Workflow
When executed:
Download the necessary NLTK resources.
Reads a CSV file (detikcom_20240601_001841.csv).
Applies preprocessing steps to text columns.
Performs sentiment analysis using a pre-trained BERT model.
Saves and visualizes the processed and labeled data.
12. Notes
Specific files like kamusalay.csv and Normalisasi.csv are required for normalization.
Pre-trained Hugging Face models are used for sentiment analysis.
Additional packages like Word Cloud and Transformers must be installed if unavailable.
