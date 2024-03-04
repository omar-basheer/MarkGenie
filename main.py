from app.utils.model_utils import print_ascii_art
print_ascii_art()

import nltk
from tests import dummy_data
from app.utils.gpt_utils import GPT3Client
from app.controllers.grading_controller import collect_input
from app.utils.grading_utils import extractive_marking, abstractive_marking
import matplotlib.pyplot as plt


def plot_marks(extractive_marks, abstractive_marks, title):
    x = range(len(extractive_marks))  # Shared x-axis for both plots
    plt.figure(figsize=(10, 6))
    plt.plot(x, extractive_marks, label="Extractive", marker='o')
    plt.plot(x, abstractive_marks, label="Abstractive", marker='x')
    plt.xlabel("Response Index")
    plt.ylabel("Mark")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_similarities(extractive_similarities, abstractive_similarities, title):
    x = range(len(extractive_similarities))  # Shared x-axis for both plots
    plt.figure(figsize=(10, 6))
    plt.plot(x, extractive_similarities, label="Extractive", marker='o')
    plt.plot(x, abstractive_similarities, label="Abstractive", marker='x')
    plt.xlabel("Response Index")
    plt.ylabel("Similarity Score")
    plt.title(title)
    plt.legend()
    plt.show()


# def main():
#     # Initialize GPT3 client if needed for abstractive marking
#     gpt_client = GPT3Client()
#     summary = gpt_client.generate_summary("break the given text into summarized points:\nWhile the log-sum-exp trick is widely used, are there potential drawbacks or limitations? Could alternative approaches be explored for specific scenarios or computational environments?")
#     print(summary)  

def main():
    print("\n >> Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize GPT3 client if needed for abstractive marking
    gpt_client = GPT3Client()

    # Define empty lists to store marks and similarity scores
    extractive_marks = []
    extractive_similarities = []
    abstractive_marks = []
    abstractive_similarities = []

    # Loop through each rubric and its responses in test_data
    for rubric, *responses in dummy_data.test_data:
        print("\nRubric:\n", rubric)

        # Mark each response using both methods
        for response in responses:
            print("\nResponse:\n", response)

            # Summarize the response using GPT3 (comment out if not using GPT3)
            generated_summary = gpt_client.generate_summary(response)
            # generated_summary = response  # Dummy data for now
            print("\nGenerated Summary:\n", generated_summary)

            print("-" * 50)
            # Extractive marking
            mark, similarity_score = extractive_marking(generated_summary, rubric)
            extractive_marks.append(mark)
            extractive_similarities.append(similarity_score)
            print(" >> Extractive Mark:", mark, "Similarity Score:", similarity_score)

            # Abstractive marking (comment out if not using GPT3)
            mark, similarity_score = abstractive_marking(generated_summary, rubric)
            abstractive_marks.append(mark)
            abstractive_similarities.append(similarity_score)
            print(" >> Abstractive Mark:", mark, "Similarity Score:", similarity_score)

            # Add a separator after each response
            print("-" * 50)
            # Create plots for marks and similarities
    plot_marks(extractive_marks, abstractive_marks, "Extractive vs. Abstractive Marks")
    # plot_similarities(extractive_similarities, abstractive_similarities, "Extractive vs. Abstractive Similarities")




# def main():
#     print(" >> Downloading NLTK resources...")
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     while True:
#         print("\n >> Collecting input...")

#         # Prompt the user for the rubric
#         rubric = input(" >> Enter the rubric (or 'exit' to quit): ")

#         if rubric.lower() == 'exit':
#             print("Exiting the program...")
#             break

#         # Prompt the user for the answer/response
#         answer = input(" >> Enter the response: ")

#         print("\n >> Initializing GPT3Client...")
#         gpt_client = GPT3Client()

#         print("\n >> Generating summary of given response...")
#         # generated_summary = gpt_client.generate_summary(answer)
#         generated_summary = dummy_data.hs_summary_answer1 # Dummy data for now

#         print("\n >> Original Response:\n", answer)
#         print("\n >> Generated Summary:\n", generated_summary)
#         print("\n >> Rubric:\n", rubric)

#         print("\n >> Marking response extractively...\n")
#         mark, similarity_score = extractive_marking(generated_summary, rubric)

#         print("\n >> Similarity Score:", similarity_score)
#         print(" >> Mark:", mark)

#         print("\n >> Marking response abstractively...\n")
#         mark, similarity_score = abstractive_marking(generated_summary, rubric)

#         print("\n >> Similarity Score:", similarity_score)
#         print(" >> Mark:", mark)

# def main():
#     print(" >> Downloading NLTK resources...")
#     nltk.download('punkt')
#     nltk.download('stopwords')

#     print("\n >> Collecting input...")
#     rubric = dummy_data.rubric1
#     answer = dummy_data.hs_summary_answer1

#     print("\n >> Initializing GPT3Client...")
#     gpt_client = GPT3Client()

#     print("\n >> Generating summary of given response...")
#     # generated_summary = gpt_client.generate_summary(answer)
#     generated_summary = dummy_data.hs_summary_answer1

#     print("\nOriginal Response:\n", answer)
#     print("\nGenerated Summary:\n", generated_summary)
#     print("\nRubric:\n", rubric)

#     print("\n >> Marking response extractively...\n")
#     mark, similarity_score = extractive_marking(generated_summary, rubric)

#     print("\n >> Similarity Score:", similarity_score)
#     print(" >> Mark:", mark)

#     print("\n >> Marking response abstractively...\n")
#     # mark, similarity_score = abstractive_marking(generated_summary, rubric, gpt_client)
#     mark, similarity_score = abstractive_marking(generated_summary, rubric)

#     print("\n >> Similarity Score:", similarity_score)
#     print(" >> Mark:", mark)

if __name__ == "__main__":
    main()

