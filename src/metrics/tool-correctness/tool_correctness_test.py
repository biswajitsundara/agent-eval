from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from dotenv import load_dotenv

load_dotenv()

test_case = LLMTestCase(
    input="Where is my order ORD-1042?",
    actual_output="Your order has shipped.",
    tools_called=[ToolCall(name="get_order_status")],      # what the agent did
    expected_tools=[ToolCall(name="get_order_status")],    # what it should have done
)

metric = ToolCorrectnessMetric()
evaluate(test_cases=[test_case], metrics=[metric])