import nltk
from app.utils.grading_utils import extractive_marking, abstractive_marking
nltk.download('stopwords')
nltk.download('punkt')
from app.controllers.grading_controller import collect_input
from app.utils.data_utils import preprocess_text
from app.utils.gpt_utils import GPT3Client
from tests import dummy_data
from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("collecting input...")
    rubric = dummy_data.rubric1
    answer = dummy_data.hs_summary_answer1

    print("initializing GPT3Client...")
    gpt_client = GPT3Client()

    print("generating summary...")
    # generated_summary = gpt_client.generate_summary(answer)
    generated_summary = dummy_data.negative_summary_answer1

    print("\nOriginal Response:\n", answer)
    print("\nGenerated Summary:\n", generated_summary)
    print("\nRubric:\n", rubric)

    print("\nmarking response extractively...\n")
    mark, similarity_score = extractive_marking(generated_summary, rubric)

    print("\nSimilarity Score:", similarity_score)
    print("Mark:", mark)

    print("\nmarking response abstractively...\n")
    # mark, similarity_score = abstractive_marking(generated_summary, rubric, gpt_client)
    mark, similarity_score = abstractive_marking(generated_summary, rubric)

    print("\nSimilarity Score:", similarity_score)
    print("Mark:", mark)

if __name__ == "__main__":
    main()

    # Step 1: Collect input
    # questions, responses, rubrics = collect_input()

    # # Step 2: Preprocess text
    # preprocessed_questions = [preprocess_text(question) for question in questions]
    # preprocessed_responses = [preprocess_text(response) for response in responses]
    # preprocessed_rubrics = [preprocess_text(rubric) for rubric in rubrics]

    # Step 3: Display preprocessed data
    # print("\nPreprocessed Questions:", preprocessed_questions)
    # print("Preprocessed Responses:", preprocessed_responses)
    # print("Preprocessed Rubrics:", preprocessed_rubrics)

    # Step 4: Generate summary
