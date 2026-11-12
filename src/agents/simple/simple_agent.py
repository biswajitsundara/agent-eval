from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def get_order_status(order_id: str) -> str:
    """Get the shipping status of an order by its ID."""
    return f"Order {order_id} has shipped and is in transit."


@tool
def get_refund_policy(category: str) -> str:
    """Get the refund policy for a product category."""
    return f"{category} can be returned within 30 days if unopened."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

support_agent = create_agent(
    model=llm,
    tools=[get_order_status, get_refund_policy],
    system_prompt="You are a support agent for ShopEase. Use the tools to answer questions.",
)