from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True,
)

good_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 30 days for a full refund.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
    ],
)

bad_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 90 days for a full refund.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
    ],
)

evaluate([good_test_case, bad_test_case], [metric])
