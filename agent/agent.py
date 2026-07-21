"""Minimal coding-agent harness for genai-coder (Ollama, OpenAI-compatible API).

The model replies in plain text with ONE JSON action object per turn (a
ReAct-style protocol -- no native tool calling, since fine-tuning + GGUF
quantization may have degraded the tool-call token template). The harness
extracts the action, executes it inside --workdir, appends the result as an
"OBSERVATION:" user message, and loops until "done" or --max-iters.

Usage:
  python agent\\agent.py --task "build X" --workdir path [--model genai-coder] [--max-iters 25]
  python agent\\agent.py --task-file agent\\demo_task.md --workdir path
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
API_KEY = "ollama"
TEMPERATURE = 0.2
OBS_LIMIT = 4000      # max chars of an observation fed back to the model
PRINT_LIMIT = 300     # max chars of action args echoed to the terminal
COMMAND_TIMEOUT = 60  # seconds
CHECK_HTTP_TIMEOUT = 15  # seconds to wait for a server to answer its URL

# Last content written per resolved path, so the observation can warn the model
# when it rewrites a file unchanged (a small model stuck in a fix-nothing loop).
LAST_WRITES = {}

# Commands containing any of these substrings (case-insensitive) are refused.
# Deliberately a dumb, visible list -- not a security sandbox, just a tripwire
# against obviously destructive commands from a confused model.
DENYLIST = [
    "rm -rf",
    "rm -fr",
    "remove-item -recurse",
    "rmdir /s",
    "del /s",
    "format-volume",
    "format c:",
    "mkfs",
    "diskpart",
    "shutdown",
    "stop-computer",
    "restart-computer",
    "reg delete",
]

SYSTEM_PROMPT = """You are genai-coder, an autonomous coding agent. You will be given a TASK.
Complete it step by step using tools. Your working directory is:
{workdir}

RULES:
- Every reply must contain EXACTLY ONE JSON action object.
- You may write one short line of reasoning before the JSON.
- For write_file, put the COMPLETE file content in ONE fenced code block
  immediately after the JSON — raw code, no JSON escaping. For every other
  tool, nothing comes after the JSON.
- All file paths are relative to the working directory. Paths outside it are rejected.
- After every action you receive an OBSERVATION message with the result. Read it before acting again.

TOOLS (use exactly these shapes):
{"tool": "list_dir", "args": {"path": "."}}
{"tool": "read_file", "args": {"path": "file.py"}}
{"tool": "write_file", "args": {"path": "file.py"}}
```
full file content here, exactly as it should appear on disk
```
{"tool": "run_command", "args": {"cmd": "python file.py"}}
{"tool": "check_http", "args": {"cmd": "python app.py", "url": "http://127.0.0.1:8005/"}}
{"tool": "done", "args": {"message": "what you built and how to run it"}}

NOTES:
- write_file overwrites the whole file, so the fenced block must hold the
  complete file content.
- run_command runs in PowerShell inside the working directory with a 60 second
  timeout. Never start long-running servers with it; verify with quick commands.
- check_http starts cmd, waits until url answers, returns the HTTP status and
  the start of the body, then stops the server. It is the ONLY correct way to
  verify a web server. Only localhost URLs are allowed.
- Verify your work with run_command (run the script, check imports) before "done".
- If run_command fails, the traceback names the exact file and line. First
  read_file that file to see what is really in it, then rewrite THAT file --
  the one named in the traceback, not another file.
- Use "done" only when the TASK is fully complete.

EXAMPLE REPLY:
I will create the script first.
{"tool": "write_file", "args": {"path": "hello.py"}}
```python
print('hi')
```
"""


def truncate(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated " + str(len(text) - limit) + " chars]"


FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_action_span(text):
    """Return (action, end_index) for the first JSON object with a "tool" key.

    end_index points just past the JSON, so callers can look for content (e.g.
    a fenced file body) that follows it. Handles bare JSON, JSON inside code
    fences, JSON preceded by prose, and (via a cleanup pass) trailing commas.
    Returns (None, None) if no action is found.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        chunk = text[i:]
        cleaned = re.sub(r",\s*([}\]])", r"\1", chunk)  # drop trailing commas
        for candidate in (chunk, cleaned):
            try:
                obj, end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "tool" in obj:
                return obj, i + end
            break  # parsed fine but not an action; keep scanning from next {
    return None, None


def extract_action(text):
    return extract_action_span(text)[0]


def fenced_content_after(text, start):
    """Return the body of the first fenced code block at/after start, else None."""
    m = FENCE_RE.search(text, start)
    return m.group(1) if m else None


def resolve_inside(workdir, path):
    """Resolve path (relative to workdir) to an absolute Path.

    Returns None if the resolved path escapes workdir.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(workdir) / p
    p = Path(os.path.realpath(p))
    root = os.path.normcase(os.path.realpath(workdir))
    child = os.path.normcase(str(p))
    if child == root or child.startswith(root + os.sep):
        return p
    return None


def path_rejected(path, workdir):
    return (
        "ERROR: path " + repr(str(path)) + " resolves outside the working directory "
        + str(workdir) + ". All paths must stay inside the working directory."
    )


def tool_read_file(args, workdir):
    raw = args.get("path")
    if not raw:
        return "ERROR: read_file needs a 'path' arg."
    path = resolve_inside(workdir, str(raw))
    if path is None:
        return path_rejected(raw, workdir)
    if not path.is_file():
        return "ERROR: file not found: " + str(raw)
    return path.read_text(encoding="utf-8", errors="replace")


def tool_write_file(args, workdir):
    raw = args.get("path")
    if not raw:
        return "ERROR: write_file needs a 'path' arg."
    if "content" not in args:
        return (
            "ERROR: write_file got no file content. Put the COMPLETE file in one"
            " fenced code block immediately after the JSON action."
        )
    path = resolve_inside(workdir, str(raw))
    if path is None:
        return path_rejected(raw, workdir)
    content = str(args.get("content", ""))
    repeat = LAST_WRITES.get(str(path)) == content
    LAST_WRITES[str(path)] = content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    msg = "Wrote " + str(len(content)) + " chars to " + str(raw)
    if repeat:
        msg += (
            "\nWARNING: this content is byte-identical to your previous write to this"
            " file. Writing the same file again cannot fix the error. Use read_file on"
            " the file and line named in the last error message, and fix that file."
        )
    return msg


def tool_list_dir(args, workdir):
    raw = args.get("path") or "."
    path = resolve_inside(workdir, str(raw))
    if path is None:
        return path_rejected(raw, workdir)
    if not path.is_dir():
        return "ERROR: not a directory: " + str(raw)
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    return "\n".join(name + ("/" if (path / name).is_dir() else "") for name in entries)


def tool_run_command(args, workdir):
    cmd = str(args.get("cmd", "")).strip()
    if not cmd:
        return "ERROR: run_command needs a 'cmd' arg."
    lowered = cmd.lower()
    for pattern in DENYLIST:
        if pattern in lowered:
            return "ERROR: command blocked by denylist pattern " + repr(pattern) + "."
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after " + str(COMMAND_TIMEOUT) + "s: " + cmd
    return (
        "exit code: " + str(proc.returncode)
        + "\nstdout:\n" + proc.stdout
        + "\nstderr:\n" + proc.stderr
    )


def tool_check_http(args, workdir):
    """Start a server command, wait for its URL to answer, report, stop it."""
    cmd = str(args.get("cmd", "")).strip()
    url = str(args.get("url", "")).strip()
    if not cmd or not url:
        return "ERROR: check_http needs 'cmd' (server command) and 'url' args."
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ("127.0.0.1", "localhost"):
        return "ERROR: check_http only accepts localhost URLs."
    lowered = cmd.lower()
    for pattern in DENYLIST:
        if pattern in lowered:
            return "ERROR: command blocked by denylist pattern " + repr(pattern) + "."
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    try:
        deadline = time.time() + CHECK_HTTP_TIMEOUT
        last_err = "no response"
        while time.time() < deadline:
            if proc.poll() is not None:
                out = (proc.stdout.read() or "").strip()
                return (
                    "ERROR: server command exited (code " + str(proc.returncode)
                    + ") before answering. output:\n" + truncate(out, 1500)
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    body = resp.read(2000).decode("utf-8", errors="replace")
                    return (
                        "HTTP " + str(resp.status) + " from " + url
                        + "\nbody:\n" + truncate(body, 800)
                        + "\n(server started, answered, and was stopped)"
                    )
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        return (
            "ERROR: " + url + " did not answer within "
            + str(CHECK_HTTP_TIMEOUT) + "s (" + last_err + ")."
        )
    finally:
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
            )


def run_tool(tool, args, workdir):
    """Execute one non-done tool action; always return an observation string."""
    try:
        if tool == "read_file":
            return tool_read_file(args, workdir)
        if tool == "write_file":
            return tool_write_file(args, workdir)
        if tool == "list_dir":
            return tool_list_dir(args, workdir)
        if tool == "run_command":
            return tool_run_command(args, workdir)
        if tool == "check_http":
            return tool_check_http(args, workdir)
        return (
            "ERROR: unknown tool " + repr(tool)
            + ". Valid tools: read_file, write_file, list_dir, run_command, check_http, done."
        )
    except Exception as e:  # keep the loop alive no matter what a tool does
        return "ERROR: " + type(e).__name__ + ": " + str(e)


def run_agent(client, model, task, workdir, max_iters, temperature=TEMPERATURE):
    """Drive the model until it says done or max_iters is hit. True if done."""
    workdir = Path(os.path.realpath(str(workdir)))
    LAST_WRITES.clear()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{workdir}", str(workdir))},
        {"role": "user", "content": "TASK:\n" + task},
    ]
    pending_write = None  # path of a write_file still waiting for its file body
    for i in range(1, max_iters + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        reply = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": reply})

        action, span_end = extract_action_span(reply)
        if action is None and pending_write is not None:
            fenced = fenced_content_after(reply, 0)
            if fenced is not None:
                # the model answered the previous content-less write_file with a
                # bare fenced block: treat it as that file's body
                action = {"tool": "write_file",
                          "args": {"path": pending_write, "content": fenced}}
        if action is None:
            print("[" + str(i) + "/" + str(max_iters) + "] no valid action in reply:")
            print("  " + truncate(reply, PRINT_LIMIT).replace("\n", "\n  "))
            obs = (
                "No valid JSON action found in your reply. Respond with exactly one "
                'JSON object, e.g. {"tool": "list_dir", "args": {"path": "."}}'
            )
        else:
            tool = action.get("tool")
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if tool == "done" and pending_write:
                # refuse to finish while a write_file never got its file body
                print(
                    "[" + str(i) + "/" + str(max_iters) + "] done rejected:"
                    " pending write_file for " + pending_write
                )
                obs = (
                    "You are not done. write_file for " + pending_write + " never"
                    " received its file content. Reply with ONE fenced code block"
                    " containing the COMPLETE contents of " + pending_write
                    + " and nothing else."
                )
                obs = truncate(obs, OBS_LIMIT)
                print("  observation: " + obs.replace("\n", "\n  "))
                messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})
                continue
            pending_write = None
            if tool == "write_file" and not str(args.get("content", "")).strip():
                # content missing or blank (some replies carry a stub "content"
                # plus the real body in a fence) -- prefer the fenced block
                fenced = fenced_content_after(reply, span_end)
                if fenced is not None:
                    args["content"] = fenced
            print(
                "[" + str(i) + "/" + str(max_iters) + "] " + str(tool) + " "
                + truncate(json.dumps(args), PRINT_LIMIT)
            )
            if tool == "done":
                print("\n[done] " + str(args.get("message", "")))
                return True
            if (tool == "write_file" and args.get("path")
                    and not str(args.get("content", "")).strip()):
                pending_write = str(args["path"])
                obs = (
                    "write_file for " + pending_write + " received no file content."
                    " Reply with ONE fenced code block containing the COMPLETE"
                    " contents of " + pending_write + " and nothing else."
                )
            else:
                obs = run_tool(tool, args, workdir)

        obs = truncate(obs, OBS_LIMIT)
        print("  observation: " + obs.replace("\n", "\n  "))
        messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})

    print("\n[stopped] reached max iterations (" + str(max_iters) + ") without done.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Minimal coding-agent harness for genai-coder.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="task description text")
    group.add_argument("--task-file", help="path to a file containing the task")
    parser.add_argument("--workdir", required=True, help="directory the agent works in (created if missing)")
    parser.add_argument("--model", default="genai-coder")
    parser.add_argument("--max-iters", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    task = args.task or Path(args.task_file).read_text(encoding="utf-8")
    os.makedirs(args.workdir, exist_ok=True)

    from openai import OpenAI  # imported here so tests never need the package

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    ok = run_agent(client, args.model, task, args.workdir, args.max_iters,
                   temperature=args.temperature)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
