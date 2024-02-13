import os
import dill
import tensorflow_hub as hub

model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"

def load_use_model(model_url):
    if os.path.exists('use_model.pkl'):
        with open('use_model.pkl', 'rb') as f:
            use_model = dill.load(f)
    else:
        use_model = hub.load(model_url)
        with open('use_model.pkl', 'wb') as f:
            dill.dump(use_model, f)
    return use_model

# def get_sentence_embedding(text, use_model):
#     embedding = use_model([text])
#     return embedding

def get_sentence_embedding(text):
    # Use the Universal Sentence Encoder to get embeddings for the input text
    use_model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")
    embedding = use_model([text]) # type: ignore
    return embedding


def print_ascii_art():
    ascii_art = """

    |  \/  | __ _ _ __| | __/ ___| ___ _ __ (_) ___
    | |\/| |/ _` | '__| |/ / |  _ / _ \ '_ \| |/ _ \\
    | |  | | (_| | |  |   <| |_| |  __/ | | | |  __/
    |_|  |_|\__,_|_|  |_|\_\\_____|\___|_| |_|_|\___|
                                                     
    """
    print(ascii_art)