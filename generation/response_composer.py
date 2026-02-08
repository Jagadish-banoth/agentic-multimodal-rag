def compose_chat_response(result: dict) -> str:
    """
    Production-grade response composer.
    Converts grounded JSON into a polished, ChatGPT-style conversational answer.
    
    Supports both:
    - Full LLM output format: {"concise_answer": ..., "answer_long": ..., ...}
    - Extractive answerer format: {"answer": ..., "reason": ...}
    """

    if not result or not isinstance(result, dict):
        return "I couldn't generate a response based on the available information."

    # Support both LLM format (concise_answer) and extractive format (answer)
    concise = (result.get("concise_answer") or result.get("answer") or "").strip()
    long_answer = (result.get("answer_long") or "").strip()
    intent = result.get("intent", "")
    confidence = result.get("confidence", {}) or {}
    next_actions = result.get("recommended_next_actions") or []
    
    # Handle extractive answerer fallback messages
    if concise == "Insufficient evidence":
        return (
            "I don't have enough information in the provided documents to answer that confidently.\n\n"
            "If you can share more context—such as the specific section, page, or figure—you want explained, "
            "I can give you a much more precise answer."
        )

    # 1️⃣ Handle insufficient context early
    if "INSUFFICIENT CONTEXT" in concise.upper():
        return (
            "I don’t have enough information in the provided documents to answer that confidently.\n\n"
            "If you can share more context—such as the specific section, page, or figure—you want explained, "
            "I can give you a much more precise answer."
        )

    response_parts = []

    # 2️⃣ Rewrite opening line to sound authoritative (VERY important)
    if concise:
        opening = concise
        if not opening.endswith("."):
            opening += "."
        response_parts.append(opening)

    # 3️⃣ Add structured explanation (ChatGPT-style flow)
    if long_answer:
        if intent in {"factual_qa", "summary"}:
            response_parts.append("\n" + long_answer)
        elif intent in {"analytical", "comparison"}:
            response_parts.append("\nHere’s how to understand this in more detail:\n\n" + long_answer)
        elif intent == "extraction":
            response_parts.append("\nBased on the document, here’s what can be extracted:\n\n" + long_answer)
        else:
            response_parts.append("\n" + long_answer)

    # 4️⃣ Confidence-aware softening (only when needed)
    if confidence.get("level") == "LOW":
        response_parts.append(
            "\n⚠️ Note: The available information is limited, so this explanation may not capture all details."
        )

    # 5️⃣ Smart follow-up suggestions (optional, capped)
    if next_actions:
        response_parts.append(
            "\nIf you want to explore this further, you could:\n- "
            + "\n- ".join(next_actions[:2])
        )

    return "\n".join(response_parts).strip()
