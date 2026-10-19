#!/usr/bin/env python3
"""Raw model text for the same tool request rendered by each chat template (no tool parser, no grammar).

    raw_example.py --base URL --export DIR --derived chat_template.jinja --out raw.json

Renders the messages with the Hub template and with the JSON-schema template (transformers), sends the prompt
string to /v1/completions (temperature 0, 96 tokens) and stores the untouched completions.
"""
import argparse
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}]
PROMPTS = ["What's the weather in Paris right now, in celsius?", "Is it raining in Oslo?",
           "Check the weather in Tokyo in fahrenheit.", "How warm is it in Lima today?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--derived", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.export, local_files_only=True)
    templates = {"hub": (args.export / "chat_template.jinja").read_text(), "json": args.derived.read_text()}
    out = {}
    for label, template in templates.items():
        rows = []
        for prompt in PROMPTS:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tools=TOOLS, chat_template=template,
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            body = {"model": "cosmos3-edge", "prompt": text, "temperature": 0, "max_tokens": 96}
            req = urllib.request.Request(args.base.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            completion = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["text"]
            rows.append({"prompt": prompt, "completion": completion})
        out[label] = {"tools_section": text.split("<tools>", 1)[1].split("</tools>", 1)[0][:1500], "rows": rows}
    args.out.write_text(json.dumps(out, indent=2))
    for label, data in out.items():
        for row in data["rows"]:
            print(label, json.dumps(row["completion"])[:200])


if __name__ == "__main__":
    main()
