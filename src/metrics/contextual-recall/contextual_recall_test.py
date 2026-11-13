from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

metric = ContextualRecallMetric(threshold=0.7, include_reason=True)

good_case = LLMTestCase(
    input="What is the return policy and how long does shipping take?",
    actual_output="You can return items within 30 days, refunds go to your original payment method, and shipping takes 3-5 business days.",
    expected_output="Returns are accepted within 30 days. Refunds go to the original payment method. Shipping takes 3-5 business days.",
    retrieval_context=[
        "All customers are eligible for a 30 day return window.",             # supports statement 1
        "Refunds are issued to the original payment method.",                 # supports statement 2
        "Standard shipping takes 3-5 business days within the country.",      # supports statement 3
    ],
)


bad_case = LLMTestCase(
    input="What is the return policy and how long does shipping take?",
    actual_output="You can return items within 30 days.",
    expected_output="Returns are accepted within 30 days. Refunds go to the original payment method. Shipping takes 3-5 business days.",
    retrieval_context=[
        "All customers are eligible for a 30 day return window.",   # supports statement 1
        "Our shoes are made from recycled materials.",              # irrelevant
        "Our store is open Mon-Sat, 9am-6pm.",                      # irrelevant
    ],
)


evaluate([good_case, bad_case], [metric])