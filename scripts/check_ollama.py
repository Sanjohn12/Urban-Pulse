"""Test the configured Ollama model with a real LangChain call, without fallback."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_workflow as workflow
from langchain_ollama import ChatOllama


def check_connection():
    available, message = workflow.check_ollama_available()
    report = {
        "base_url": workflow.OLLAMA_BASE_URL,
        "model": workflow.OLLAMA_MODEL,
        "model_list_available": available,
        "availability_message": message,
        "real_llm_call_succeeded": False,
    }
    try:
        # Same adapter/model/endpoint as the app; bounded diagnostic output/time.
        llm = ChatOllama(
            model=workflow.OLLAMA_MODEL, base_url=workflow.OLLAMA_BASE_URL,
            temperature=0, num_predict=64, client_kwargs={"timeout": 30},
        )
        response = llm.invoke("Reply with the word READY. /no_think")
        content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        if not content:
            raise ValueError("Ollama returned no visible response")
        report.update(real_llm_call_succeeded=True, response=content)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    report = check_connection()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["real_llm_call_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
