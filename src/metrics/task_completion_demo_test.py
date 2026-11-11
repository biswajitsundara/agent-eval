from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
load_dotenv()

metric = TaskCompletionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True)

# Case 1: the output fully accomplishes the task
good_case = LLMTestCase(
    input="Write a 2-sentence summary of what photosynthesis is.",
    actual_output=(
        "Photosynthesis is the process by which plants, algae and some bacteria "
        "convert light energy into chemical energy stored as glucose. "
        "It uses carbon dioxide and water, and releases oxygen as a by-product."
    ),
)

# Case 2: the output only partly does what was asked
bad_case = LLMTestCase(
    input="Write a 2-sentence summary of what photosynthesis is, and give one real-world example.",
    actual_output="Photosynthesis is how plants make food from sunlight.",
)

evaluate(test_cases=[good_case, bad_case], metrics=[metric])

# Or one at a time:
# metric.measure(bad_case)
# print(metric.score, metric.reason)