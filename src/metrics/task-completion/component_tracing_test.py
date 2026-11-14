import sys

from deepeval.contextvars import get_current_golden
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import TaskCompletionMetric
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from pathlib import Path
from deepeval.dataset import EvaluationDataset, Golden

# Go up 3 levels: task_completion_test.py -> metrics -> src -> agent-eval
project_root = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(project_root))

from src.agents.shopease.agent_instrumented import support_agent as _support_agent


#Trace have expected values and actual values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace( expected_tools=golden.expected_tools )
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output )

    return _support_agent( user_input )



task_completion = TaskCompletionMetric(threshold=0.7,model="gpt-4o-mini")
tool_correctness = ToolCorrectnessMetric()

dataSet = EvaluationDataset(goldens= [
    Golden(input = "Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")],
           expected_output="Order ORD-1042 has shipped and is in transit.",),

    Golden(input = "What is refund policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")],
           expected_output="Refunds for electronics are accepted within 30 days of purchase.")
])


for golden in dataSet.evals_iterator(metrics =[task_completion,tool_correctness]):
    support_agent(golden.input)