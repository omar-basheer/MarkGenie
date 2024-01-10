
def collect_input():
    """
    Collects input data, including questions, student responses, and rubrics.

    Returns:
    - questions (list): List of open-ended questions.
    - responses (list): List of student responses.
    - rubrics (list): List of rubrics.
    """
    questions = input("Enter open-ended questions (comma-separated): ").split(',')
    responses = input("Enter student responses (comma-separated): ").split(',')
    rubrics = input("Enter rubrics (comma-separated): ").split(',')

    return questions, responses, rubrics
