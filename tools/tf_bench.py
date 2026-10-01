"""Speed + quality gate for the TP4 GLM-5.3 endpoint (no auth, loopback). usage: tf_bench.py URL"""
import json, sys, time, urllib.request

URL = sys.argv[1].rstrip("/")


def chat(body, timeout=1800):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t


def ttft_and_rate(body):
    """Streamed: time to first token and decode rate from the chunk timestamps."""
    body = dict(body, stream=True)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    text = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                n += 1
                text += piece
                if first is None:
                    first = time.time()
    end = time.time()
    return first - t0, n, (n - 1) / max(end - first, 1e-6), text


print("== decode speed (3 runs, ~400 tokens, thinking off)")
for i in range(3):
    ttft, n, rate, text = ttft_and_rate({"messages": [{"role": "user", "content":
        "Write a detailed explanation of how a hash table works, including collisions, load factor and resizing."}],
        "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    print(f"run {i + 1}: ttft {ttft:.2f}s, {n} chunks, {rate:.2f} tok/s")
print("  sample:", repr(text[:240]))

print("== thinking on: reasoning + answer")
b, dt = chat({"messages": [{"role": "user", "content": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Answer in one sentence."}],
              "max_tokens": 2048, "temperature": 0})
m = b["choices"][0]["message"]
print(f"{dt:.1f}s usage={b['usage']['completion_tokens']} finish={b['choices'][0]['finish_reason']}")
print("  reasoning:", repr((m.get("reasoning_content") or "")[:200]))
print("  content:", repr(m.get("content")))

print("== long prompt (prefill)")
filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(900))
needle = "The secret code word is PELICAN-42."
doc = filler[:len(filler) // 2] + " " + needle + " " + filler[len(filler) // 2:]
b, dt = chat({"messages": [{"role": "user", "content": doc + "\n\nWhat is the secret code word? Reply with just the code word."}],
              "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
print(f"{dt:.1f}s prompt={b['usage']['prompt_tokens']} tok -> {b['usage']['prompt_tokens'] / dt:.0f} tok/s incl. decode;"
      f" answer={b['choices'][0]['message'].get('content')!r}")

print("== tool call")
b, dt = chat({"messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
              "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
              "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = b["choices"][0]["message"]
print(f"{dt:.1f}s finish={b['choices'][0]['finish_reason']} tool_calls={json.dumps(m.get('tool_calls'))[:200]} content={m.get('content')!r}")
