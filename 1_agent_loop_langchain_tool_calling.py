from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# Tools


@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    # Simulate fetching product price from a database or API
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    print(f"Fetching price for product: {product}")
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price. Available for bronze, silver, and gold tiers."""
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    print(f"Applying {discount_percentage}% discount for tier: {discount_tier}")
    return price * (1 - discount_percentage / 100)


# Agent Loop
@traceable(
    name="Langchain Agent Loop",
    description="An agent that can call tools to get product prices and apply discounts.",
)
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
        model=f"ollama:{MODEL}",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {query}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=query),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration} of {MAX_ITERATIONS}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"AI Response: {ai_message.content}")
            return ai_message.content

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            tool_to_use = tools_dict.get(tool_name)

            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' is not available.")

            observation = tool_to_use.invoke(tool_args)
            messages.append(
                ToolMessage(tool_call_id=tool_call_id, content=str(observation))
            )


if __name__ == "__main__":
    run_agent("I want to buy a laptop and apply a gold discount.")
