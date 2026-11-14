# STAGE 6 — THE TOOL
# ---------------------------------------------------------------------------
vector_store = None
retriever = (
    vector_store.as_retriever(search_kwargs={"k": 3})
    if vector_store is not None
    else None
)


def search_policy(question: str) -> str:
    """Search the company's Inventory & Sales policy documents for information
    about returns, bulk discounts, supplier terms, restocking, complaints,
    shipping/delivery, warranties and repairs, customer credit terms, price
    matching, damaged or lost shipments, stocktakes and shrinkage, gift cards
    and store credit, or special orders and backorders."""
    matching_chunks = retriever.invoke(question)
    return "\n\n".join(chunk.page_content for chunk in matching_chunks)