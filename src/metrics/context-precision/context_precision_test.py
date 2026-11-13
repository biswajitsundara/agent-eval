from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4o-mini",       # the LLM judge
    include_reason=True,  # returns a natural-language explanation
)

good_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 30 days for a full refund.",
    expected_output="Customers can return shoes within 30 days for a full refund.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Our shoes are made from recycled materials.",
    ],
)



bad_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 30 days for a full refund.",
    expected_output="Customers can return shoes within 30 days for a full refund.",
    retrieval_context=[
        "Our shoes are made from recycled materials.",                         # irrelevant
        "We ship to over 50 countries.",                                       # irrelevant
        "All customers are eligible for a 30 day full refund at no extra cost.",  # relevant
        "Our store is open Mon-Sat, 9am-6pm.",                                 # irrelevant
    ],
)

evaluate([good_test_case, bad_test_case], [metric])