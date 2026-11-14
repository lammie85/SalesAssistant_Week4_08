# STAGE 7 — THE AGENT
# ---------------------------------------------------------------------------

def create_agent(model, tools, system_prompt):
    """Compatibility shim for the inventory policy agent."""
    return {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
    }


def search_policy(query):
    """Placeholder policy lookup tool.

    The real implementation should search the policy corpus and return the
    relevant results for the supplied query.
    """
    return []


SYSTEM_PROMPT = (
    "You are a friendly Inventory & Sales policy assistant. Always use the "
    "search_policy tool to look up facts before answering. If the answer "
    "isn't in the search results, say you don't know. Never promise a "
    "refund directly — always say it needs human approval, per policy."
)

inventory_assistant = create_agent(
    model=None,
    tools=[search_policy],
    system_prompt=SYSTEM_PROMPT,
)

print("Inventory & Sales assistant is ready.\n")