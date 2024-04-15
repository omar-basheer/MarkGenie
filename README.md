# MarkGenie

## Overview
Mark Genie is a tool for automatically grading student responses based on their similarity to a provided rubric. It offers both extractive and abstractive marking methods 
to evaluate the similarity between the a student's response and a given rubric. The tool leverages natural language processing techniques and 
machine learning models to automate the grading process.

## How it Works
Mark Genie utilizes natural language processing (NLP) techniques to analyze and compare student responses with a predefined rubric. Each student response is 
first passed to openai's gpt api to be summarized. The summarized responses are then compared to the rubric via either extractive or abstractive marking.

1. Extractive Marking: This method involves vectorizing the rubric and the summarized student response using TF-IDF (Term Frequency-Inverse Document Frequency) vectors. 
The Cosine similarity is then calculated between the vectors to determine the similarity score.

2. Abstractive Marking: In this method, the Universal Sentence Encoder (USE) provided by Google is used to encode the semantic meaning of both the rubric 
and the summarized student response into fixed-length embeddings. The Cosine similarity is then computed between the embeddings to determine the similarity score.

The similarity scores obtained from either method is mapped to a mark using predefined thresholds.

## Technologies Used
* Python: The core programming language used for developing Mark Genie.
* Natural Language Toolkit (NLTK): Used for text preprocessing and tokenization.
* TensorFlow: Used for loading and utilizing the Universal Sentence Encoder.
* scikit-learn: Used for implementing the TF-IDF vectorizer and cosine similarity calculations.
* Chat GPT api: Used for summarization of each student response.
