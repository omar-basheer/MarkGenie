from app.utils.model_utils import print_ascii_art
print_ascii_art()

import nltk
print(" >> Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')

from tests import dummy_data
from app.utils.gpt_utils import GPT3Client
from app.controllers.grading_controller import collect_input
from app.utils.grading_utils import extractive_marking, abstractive_marking


def main():
    while True:
        print("\n >> Collecting input...")

        # Prompt the user for the rubric
        rubric = input(" >> Enter the rubric (or 'exit' to quit): ")

        if rubric.lower() == 'exit':
            print("Exiting the program...")
            break

        # Prompt the user for the answer/response
        answer = input(" >> Enter the response: ")

        print("\n >> Initializing GPT3Client...")
        gpt_client = GPT3Client()

        print("\n >> Generating summary of given response...")
        # generated_summary = gpt_client.generate_summary(answer)
        generated_summary = dummy_data.hs_summary_answer1 # Dummy data for now

        print("\n >> Original Response:\n", answer)
        print("\n >> Generated Summary:\n", generated_summary)
        print("\n >> Rubric:\n", rubric)

        print("\n >> Marking response extractively...\n")
        mark, similarity_score = extractive_marking(generated_summary, rubric)

        print("\n >> Similarity Score:", similarity_score)
        print(" >> Mark:", mark)

        print("\n >> Marking response abstractively...\n")
        mark, similarity_score = abstractive_marking(generated_summary, rubric)

        print("\n >> Similarity Score:", similarity_score)
        print(" >> Mark:", mark)

# def main():
#     print("\ncollecting input...")
#     rubric = dummy_data.rubric1
#     answer = dummy_data.hs_summary_answer1

#     print("initializing GPT3Client...")
#     gpt_client = GPT3Client()

#     print("generating summary...")
#     # generated_summary = gpt_client.generate_summary(answer)
#     generated_summary = dummy_data.hs_summary_answer1

#     print("\nOriginal Response:\n", answer)
#     print("\nGenerated Summary:\n", generated_summary)
#     print("\nRubric:\n", rubric)

#     print("\nmarking response extractively...\n")
#     mark, similarity_score = extractive_marking(generated_summary, rubric)

#     print("\nSimilarity Score:", similarity_score)
#     print("Mark:", mark)

#     print("\nmarking response abstractively...\n")
#     # mark, similarity_score = abstractive_marking(generated_summary, rubric, gpt_client)
#     mark, similarity_score = abstractive_marking(generated_summary, rubric)

#     print("\nSimilarity Score:", similarity_score)
#     print("Mark:", mark)

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
