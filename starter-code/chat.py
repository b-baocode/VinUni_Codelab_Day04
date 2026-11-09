"""
Chat tương tác với VinAssistant trên terminal.

  - Có GEMINI_API_KEY + đã cài google-genai  → chat với Gemini thật, model tự quyết định gọi tool.
  - Ngược lại                               → chat với ToolCallingAgent (mock, keyword/regex).

Chạy:
    python chat.py            # tự chọn chế độ
    python chat.py --mock     # ép dùng mock agent

Lệnh trong lúc chat: /trace (xem trace bước gần nhất), /reset (xoá lịch sử), /exit (thoát).
"""

import json
import os
import sys
from typing import Any, Dict, List

from template import ToolCallingAgent, build_system_prompt
from tools import TOOL_DEFINITIONS, TOOL_MAP

MAX_ITERATIONS = 5
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


# ═══════════════════════════════════════════════════════════════════════════
# Chế độ 1: Mock — dùng lại ToolCallingAgent của bài lab
# ═══════════════════════════════════════════════════════════════════════════

class MockChat:
    name = "mock (ToolCallingAgent)"

    def __init__(self):
        self.agent = ToolCallingAgent(max_iterations=MAX_ITERATIONS)
        self.last_trace: List[Dict[str, Any]] = []

    def send(self, user_input: str) -> str:
        result = self.agent.run(user_input)
        self.last_trace = result["trace"]
        return result["answer"]

    def reset(self):
        self.last_trace = []


# ═══════════════════════════════════════════════════════════════════════════
# Chế độ 2: Gemini thật — function calling với TOOL_DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class GeminiChat:
    def __init__(self, api_key: str):
        from google import genai
        from google.genai import types

        self.types = types
        # timeout (ms) để không treo vô hạn khi model quá tải
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
        self.name = f"Gemini ({GEMINI_MODEL})"
        self.config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(MAX_ITERATIONS),
            # TOOL_DEFINITIONS đã đúng format {name, description, parameters}
            tools=[types.Tool(function_declarations=TOOL_DEFINITIONS)],
            # Tự xử lý vòng lặp tool để còn ghi trace
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.history: List[Any] = []
        self.last_trace: List[Dict[str, Any]] = []

    def send(self, user_input: str) -> str:
        types = self.types
        self.last_trace = []
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

        for iteration in range(1, MAX_ITERATIONS + 1):
            response = self.client.models.generate_content(
                model=GEMINI_MODEL, contents=self.history, config=self.config
            )
            model_content = response.candidates[0].content
            self.history.append(model_content)

            calls = response.function_calls or []
            if not calls:  # Không gọi tool nữa → Final Answer
                return response.text or "(model không trả lời)"

            # Model yêu cầu gọi tool → thực thi và gửi Observation lại
            result_parts = []
            for call in calls:
                args = dict(call.args or {})
                try:
                    observation = TOOL_MAP[call.name](**args)
                except Exception as e:
                    observation = {"error": str(e)}
                self.last_trace.append({
                    "iteration": iteration,
                    "action": call.name,
                    "action_input": args,
                    "observation": observation,
                })
                result_parts.append(types.Part.from_function_response(
                    name=call.name, response={"result": observation}
                ))
            self.history.append(types.Content(role="user", parts=result_parts))

        return "Lỗi: Vượt quá số bước tối đa."

    def reset(self):
        self.history = []
        self.last_trace = []


# ═══════════════════════════════════════════════════════════════════════════
# REPL
# ═══════════════════════════════════════════════════════════════════════════

def make_chat(force_mock: bool):
    api_key = os.environ.get("GEMINI_API_KEY")
    if force_mock or not api_key:
        return MockChat()
    try:
        return GeminiChat(api_key)
    except ImportError:
        print("⚠️  Chưa cài google-genai (pip install google-genai) → dùng mock agent.\n")
        return MockChat()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    chat = make_chat(force_mock="--mock" in sys.argv)
    print(f"VinAssistant — chế độ: {chat.name}")
    print("Gõ /trace, /reset hoặc /exit.\n")

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            chat.reset()
            print("(đã xoá lịch sử)\n")
            continue
        if user_input == "/trace":
            print(json.dumps(chat.last_trace, indent=2, ensure_ascii=False, default=str), "\n")
            continue

        try:
            answer = chat.send(user_input)
        except Exception as e:
            answer = f"[Lỗi khi gọi model] {type(e).__name__}: {e}"
        print(f"VinAssistant: {answer}\n")


if __name__ == "__main__":
    main()
