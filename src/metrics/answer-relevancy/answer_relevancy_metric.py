from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()  

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True,
)

good_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 30 days for a full refund.",
)

bad_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="You can return them within 30 days. Our shoes are made from recycled materials. We were founded in 1998 and have stores in 12 countries.",
)

evaluate([good_case, bad_case], [metric])