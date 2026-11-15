"""Gets the draft answer from the company LLM (GPT-Neo hosted on Kaggle, see kaggle/)."""
import os
from llm import remote_generate


def _messages_to_prompt(messages):
    system = " ".join(m["content"] for m in messages
                      if m.get("role") == "system" and isinstance(m.get("content"), str))
    turns = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            turns.append(("User" if m["role"] == "user" else "Assistant") + ": " + m["content"])
    # the server appends the final "Assistant:" itself, so drop a trailing prefix-less user turn format
    last_user = turns.pop() if turns and turns[-1].startswith("User: ") else ""
    history = "\n".join(turns)
    prompt = (history + "\n" if history else "") + last_user.removeprefix("User: ")
    return system, prompt


def call_upstream(body):
    """Return (status_code, chat-completion-shaped dict). Streaming is off: the full answer must be seen first."""
    model = os.environ.get("UPSTREAM_MODEL", "gpt-neo")
    system, prompt = _messages_to_prompt(body.get("messages", []))
    try:
        text = remote_generate(model, prompt, system,
                               temperature=body.get("temperature") or 0.0,
                               max_new_tokens=body.get("max_tokens") or 200)
    except Exception as e:
        return 502, {"error": {"message": f"model server unreachable: {e}"}}
    return 200, {"id": "chatcmpl-kaggle", "object": "chat.completion", "model": model,
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": text}}]}
