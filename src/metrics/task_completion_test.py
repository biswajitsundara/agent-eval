import sys

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
import sys
from pathlib import Path

# Go up 3 levels: task_completion_test.py -> metrics -> src -> agent-eval
project_root = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_root))

from src.agents.shopease.agent_instrumented import support_agent

actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)

evaluate(test_cases = [test_case],
         metrics= [TaskCompletionMetric(threshold=0.7,model = "gpt-4o-mini")])


## Run the test
# Come to this directory : src\metrics
# Run the following command in terminal: uv run .\task_completion_test.py