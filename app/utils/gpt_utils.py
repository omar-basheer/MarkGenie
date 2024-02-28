from multiprocessing import context
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


class GPT3Client:
    def __init__(self):
        self.client = OpenAI(api_key='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')
        self.nlp = spacy.load("en_core_web_sm")

    def generate_summary(self, content):
        # response = self.client.completions.create(
        # model="gpt-3.5-turbo-instruct",
        # prompt=content,
        # # prompt="break the given text into summarized points:\nWhile the log-sum-exp trick is widely used, are there potential drawbacks or limitations? Could alternative approaches be explored for specific scenarios or computational environments?break the given text into summarized points:\nWhile the log-sum-exp trick is widely used, are there potential drawbacks or limitations? Could alternative approaches be explored for specific scenarios or computational environments?\n\n- Log-sum-exp trick is commonly used.\n- Are there any drawbacks or limitations?\n- Possible alternative approaches for certain scenarios or computational environments.",
        # temperature=1,
        # max_tokens=256,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
        # )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "break the given text into summarized points"},
                {"role": "user", "content": content},
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        generated_summary = response.choices[0].message.content
        # generated_summary = response.choices[0].text
        return generated_summary
    
    def get_word_embeddings(self, text):
        doc = self.nlp(text)
        return [token.vector for token in doc]


