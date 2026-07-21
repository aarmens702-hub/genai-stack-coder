"""Protocol-layer tests for agent.py. No network: the model client is faked.

Run: .venv\\Scripts\\python.exe agent\\test_agent.py
"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent


class FakeClient:
    """Stands in for openai.OpenAI: returns scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.replies:
            reply = self.replies.pop(0)
        else:
            reply = json.dumps({"tool": "done", "args": {"message": "fallback done"}})
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_fenced():
    text = 'Here is the action:\n```json\n{"tool": "list_dir", "args": {"path": "."}}\n```\nDone.'
    action = agent.extract_action(text)
    assert action == {"tool": "list_dir", "args": {"path": "."}}


def test_extract_prose_prefix():
    text = 'I think {maybe} we should read it.\n{"tool": "read_file", "args": {"path": "a.txt"}}'
    action = agent.extract_action(text)
    assert action == {"tool": "read_file", "args": {"path": "a.txt"}}


def test_extract_trailing_comma():
    text = '{"tool": "done", "args": {"message": "all good",},}'
    action = agent.extract_action(text)
    assert action is not None
    assert action["tool"] == "done"
    assert action["args"]["message"] == "all good"


def test_extract_nothing():
    assert agent.extract_action("no action here, just { broken text") is None
    assert agent.extract_action('{"not_a_tool": 1}') is None


def test_path_escape_rejected():
    with tempfile.TemporaryDirectory() as wd:
        assert agent.resolve_inside(wd, "..\\evil.txt") is None
        assert agent.resolve_inside(wd, "C:\\Windows\\evil.txt") is None
        sibling = os.path.realpath(wd) + "x"
        assert agent.resolve_inside(wd, os.path.join(sibling, "f.txt")) is None
        assert agent.resolve_inside(wd, "sub/ok.txt") is not None

        obs = agent.run_tool("write_file", {"path": "../evil.txt", "content": "x"}, wd)
        assert "outside the working directory" in obs
        assert not (Path(os.path.realpath(wd)).parent / "evil.txt").exists()


def test_write_read_roundtrip():
    with tempfile.TemporaryDirectory() as wd:
        content = "line one\nline two\n"
        obs = agent.run_tool("write_file", {"path": "sub/notes.txt", "content": content}, wd)
        assert obs.startswith("Wrote")
        obs2 = agent.run_tool("read_file", {"path": "sub/notes.txt"}, wd)
        assert obs2 == content


def test_run_command_captures_output():
    with tempfile.TemporaryDirectory() as wd:
        obs = agent.run_tool("run_command", {"cmd": "echo hello-from-ps"}, wd)
        assert "exit code: 0" in obs
        assert "hello-from-ps" in obs
        obs2 = agent.run_tool("run_command", {"cmd": "exit 7"}, wd)
        assert "exit code: 7" in obs2


def test_run_command_denylist():
    with tempfile.TemporaryDirectory() as wd:
        obs = agent.run_tool("run_command", {"cmd": "rm -rf /"}, wd)
        assert "blocked by denylist" in obs
        obs2 = agent.run_tool("run_command", {"cmd": "Remove-Item -Recurse C:\\stuff"}, wd)
        assert "blocked by denylist" in obs2


def test_done_ends_loop():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            json.dumps({"tool": "done", "args": {"message": "finished"}}),
            json.dumps({"tool": "list_dir", "args": {"path": "."}}),
        ])
        ok = agent.run_agent(client, "fake-model", "say done", wd, max_iters=5)
        assert ok is True
        assert client.calls == 1


def test_max_iters_stops():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([json.dumps({"tool": "list_dir", "args": {"path": "."}})] * 10)
        ok = agent.run_agent(client, "fake-model", "loop forever", wd, max_iters=3)
        assert ok is False
        assert client.calls == 3


def test_scripted_session():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            "Creating the file.\n"
            + json.dumps({"tool": "write_file",
                          "args": {"path": "hello.py", "content": "print('hi')\n"}}),
            json.dumps({"tool": "read_file", "args": {"path": "hello.py"}}),
            json.dumps({"tool": "done", "args": {"message": "built hello.py"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "make hello.py", wd, max_iters=10)
        assert ok is True
        assert client.calls == 3
        written = (Path(wd) / "hello.py").read_text(encoding="utf-8")
        assert written == "print('hi')\n"
        # the harness fed results back as OBSERVATION user messages
        sent = client.last_kwargs["messages"]
        obs_msgs = [m for m in sent if m["role"] == "user" and m["content"].startswith("OBSERVATION:")]
        assert len(obs_msgs) == 2  # one per executed action before done


def test_fenced_write():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            'Writing the file.\n'
            '{"tool": "write_file", "args": {"path": "app.py"}}\n'
            '```python\n'
            'print("/")\n'
            'x = 1\n'
            '```',
            json.dumps({"tool": "done", "args": {"message": "d"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=5)
        assert ok is True
        written = (Path(wd) / "app.py").read_text(encoding="utf-8")
        assert written == 'print("/")\nx = 1\n'


def test_write_without_content_prompts_for_fence():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            '{"tool": "write_file", "args": {"path": "app.py"}}',  # no fence, no content
            '```python\npass\n```',                                # body after the prompt
            json.dumps({"tool": "done", "args": {"message": "d"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=5)
        assert ok is True
        sent = client.last_kwargs["messages"]
        obs = [m["content"] for m in sent
               if m["role"] == "user" and m["content"].startswith("OBSERVATION:")]
        assert "received no file content" in obs[0]
        assert (Path(wd) / "app.py").read_text(encoding="utf-8") == "pass\n"


def test_split_fenced_write():
    """JSON action in one reply, bare fenced file body in the next."""
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            '{"tool": "write_file", "args": {"path": "app.py"}}',
            '```python\nprint("/")\n```',
            json.dumps({"tool": "done", "args": {"message": "d"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=5)
        assert ok is True
        written = (Path(wd) / "app.py").read_text(encoding="utf-8")
        assert written == 'print("/")\n'
        sent = client.last_kwargs["messages"]
        obs = [m["content"] for m in sent
               if m["role"] == "user" and m["content"].startswith("OBSERVATION:")]
        assert "Wrote" in obs[1]


def test_blank_content_prefers_fence():
    """A stub "content" like "\\r\\n" must not clobber the fenced body."""
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            '{"tool": "write_file", "args": {"path": "a.py", "content": "\\r\\n"}}\n'
            '```python\nx = 1\n```',
            json.dumps({"tool": "done", "args": {"message": "d"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=5)
        assert ok is True
        written = (Path(wd) / "a.py").read_text(encoding="utf-8")
        assert written == "x = 1\n"


def test_check_http_happy():
    with tempfile.TemporaryDirectory() as wd:
        (Path(wd) / "index.html").write_text("hello-check-http", encoding="utf-8")
        obs = agent.run_tool(
            "check_http",
            {"cmd": "python -m http.server 8123 --bind 127.0.0.1",
             "url": "http://127.0.0.1:8123/index.html"},
            wd,
        )
        assert "HTTP 200" in obs
        assert "hello-check-http" in obs


def test_check_http_rejects_remote():
    with tempfile.TemporaryDirectory() as wd:
        obs = agent.run_tool(
            "check_http",
            {"cmd": "python -m http.server 8124", "url": "http://example.com/"},
            wd,
        )
        assert "only accepts localhost" in obs


def test_check_http_404_is_an_answer():
    with tempfile.TemporaryDirectory() as wd:
        obs = agent.run_tool(
            "check_http",
            {"cmd": "python -m http.server 8126 --bind 127.0.0.1",
             "url": "http://127.0.0.1:8126/missing.html"},
            wd,
        )
        assert "HTTP 404" in obs
        assert "answered" in obs


def test_check_http_dead_server():
    with tempfile.TemporaryDirectory() as wd:
        obs = agent.run_tool(
            "check_http",
            {"cmd": "python -c \"print('bye')\"", "url": "http://127.0.0.1:8125/"},
            wd,
        )
        assert "ERROR" in obs
        assert "exited" in obs or "did not answer" in obs


def test_done_rejected_while_write_pending():
    with tempfile.TemporaryDirectory() as wd:
        client = FakeClient([
            '{"tool": "write_file", "args": {"path": "a.py"}}',   # no body yet
            '{"tool": "done", "args": {"message": "premature"}}',  # must be vetoed
            '```python\nx = 2\n```',                               # the body arrives
            json.dumps({"tool": "done", "args": {"message": "d"}}),
        ])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=6)
        assert ok is True
        assert client.calls == 4
        written = (Path(wd) / "a.py").read_text(encoding="utf-8")
        assert written == "x = 2\n"


def test_repeated_write_warns():
    with tempfile.TemporaryDirectory() as wd:
        same = json.dumps({"tool": "write_file",
                           "args": {"path": "a.py", "content": "x = await f()\n"}})
        client = FakeClient([same, same, json.dumps({"tool": "done", "args": {"message": "d"}})])
        ok = agent.run_agent(client, "fake-model", "write it", wd, max_iters=5)
        assert ok is True
        sent = client.last_kwargs["messages"]
        obs = [m["content"] for m in sent
               if m["role"] == "user" and m["content"].startswith("OBSERVATION:")]
        assert len(obs) == 2
        assert "byte-identical" not in obs[0]
        assert "byte-identical" in obs[1]


def test_truncation():
    long_obs = "x" * 5000
    out = agent.truncate(long_obs, agent.OBS_LIMIT)
    assert len(out) < 5000
    assert "truncated" in out


TESTS = [
    test_extract_fenced,
    test_extract_prose_prefix,
    test_extract_trailing_comma,
    test_extract_nothing,
    test_path_escape_rejected,
    test_write_read_roundtrip,
    test_run_command_captures_output,
    test_run_command_denylist,
    test_done_ends_loop,
    test_max_iters_stops,
    test_scripted_session,
    test_fenced_write,
    test_write_without_content_prompts_for_fence,
    test_split_fenced_write,
    test_blank_content_prefers_fence,
    test_check_http_happy,
    test_check_http_rejects_remote,
    test_check_http_404_is_an_answer,
    test_check_http_dead_server,
    test_done_rejected_while_write_pending,
    test_repeated_write_warns,
    test_truncation,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
            print("PASS " + test.__name__)
        except Exception:
            failures += 1
            print("FAIL " + test.__name__)
            traceback.print_exc()
    print()
    print(str(len(TESTS) - failures) + "/" + str(len(TESTS)) + " tests passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
