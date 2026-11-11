from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from dotenv import load_dotenv
load_dotenv()

metric = TaskCompletionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True)

test_case = LLMTestCase(
    input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
    actual_output="Day 1: Louvre, Seine walk, dinner at a bistro in Le Marais. "
    "Day 2: Musée d'Orsay, Montmartre, crêpes and local wine bars. "
    "Day 3: Versailles, then a classic French dinner.",
    tools_called=[ToolCall(name="Restaurant Finder", input_parameters={"city": "Paris"})],
)

# metric.measure(test_case)
# print(metric.score, metric.reason)



task = "Cancel my order #4521 and confirm the refund."

# Case 1: the tool matches the claim
good_case = LLMTestCase(
    input=task,
    actual_output="Order #4521 is cancelled and a refund of $42 has been initiated.",
    tools_called=[ToolCall(
        name="Cancel Order",
        input_parameters={"order_id": "4521"},
        output={"status": "cancelled", "refund": "initiated", "amount": 42},
    )],
)

# Case 2: the agent calls the wrong tool but still claims success
bad_case = LLMTestCase(
    input=task,
    actual_output="Order #4521 is cancelled and a refund of $42 has been initiated.",
    tools_called=[ToolCall(
        name="Find Order Status",
        input_parameters={"order_id": "4521"},
        output={"status": "shipped"},
    )],
)


evaluate(test_cases=[good_case, bad_case], metrics=[metric])
# or: evaluate([test_case], [metric])

# metric.measure(bad_case)
# print(metric.score, metric.reason)