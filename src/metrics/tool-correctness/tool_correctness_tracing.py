from deepeval.dataset import EvaluationDataset, Golden, get_current_golden  # adjust if your version differs
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from deepeval.integrations.langchain import CallbackHandler

project_root = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(project_root))

from src.agents.simple.simple_agent import support_agent as agent  # Import the support_agent from simple_agent.py

deepeval_callback = CallbackHandler()

@observe(name="support_agent")
def run_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_tools:
        update_current_trace(expected_tools=golden.expected_tools)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )

    reply = result["messages"][-1].content
    return reply

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input="What is the refund policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")]),
])

if __name__ == "__main__":
    for golden in dataset.evals_iterator(metrics=[ToolCorrectnessMetric()]):
        run_agent(golden.input)