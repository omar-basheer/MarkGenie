import numpy as np
import tensorflow_hub as hub
from app.utils.gpt_utils import GPT3Client
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, GPT2Tokenizer, GPT2Model
from app.utils.model_utils import get_sentence_embedding, load_use_model, model_url

# use_model = load_use_model(model_url)

def abstractive_marking(generated_summary, rubric):
    # Get sentence embeddings for the generated summary and rubric
    print("   * getting summary sentence embeddings...")
    generated_summary_embedding = get_sentence_embedding(generated_summary)
    print("   * getting rubric sentence embeddings...")
    rubric_embedding = get_sentence_embedding(rubric)

    # Calculate cosine similarity between embeddings
    print("   * calculating cosine similarity...")
    similarity_score = cosine_similarity(generated_summary_embedding, rubric_embedding)[0][0]

    # Map similarity score to a mark (assuming a linear scale from 1 to 10)
    print("   * mapping similarity score to mark...")
    mark = int(similarity_score * 10)
    return mark, similarity_score

def extractive_marking(generated_summary, rubric):
    # Initialize the rubric vectorizer and fit it
    rubric_vectorizer = TfidfVectorizer()
    rubric_vectorizer.fit([rubric])

    # Transform the generated summary using the rubric's fitted vectorizer
    print("   * transforming generated summary using rubric's fitted vectorizer...")
    generated_summary_vector = rubric_vectorizer.transform([generated_summary])

    # Calculate cosine similarity between generated summary and rubric
    print("   * calculating cosine similarity...")
    similarity_score = cosine_similarity(generated_summary_vector, rubric_vectorizer.transform([rubric]))[0][0]

    # Map similarity score to a mark (assuming a linear scale from 1 to 10)
    print("   * mapping similarity score to mark...")
    mark = int(similarity_score * 10)
    return mark, similarity_score


# def abstractive_marking(generated_summary, rubric, gpt_client):
#     # Generate embeddings for the rubric
#     rubric_embeddings = np.array(gpt_client.get_word_embeddings(rubric))

#     # Generate embeddings for the generated summary
#     generated_summary_embeddings = np.array(gpt_client.get_word_embeddings(generated_summary))

#     # Reshape arrays to 2D if they are 1D
#     if len(rubric_embeddings.shape) == 1:
#         rubric_embeddings = rubric_embeddings.reshape(1, -1)

#     if len(generated_summary_embeddings.shape) == 1:
#         generated_summary_embeddings = generated_summary_embeddings.reshape(1, -1)

#     # Calculate cosine similarity between embeddings
#     print(" * calculating cosine similarity...")
#     similarity_score = cosine_similarity(generated_summary_embeddings, rubric_embeddings)[0][0]

#     # Map similarity score to a mark (assuming a linear scale from 1 to 10)
#     print(" * mapping similarity score to mark...")
#     mark = int(similarity_score * 10)
#     return mark, similarity_score



