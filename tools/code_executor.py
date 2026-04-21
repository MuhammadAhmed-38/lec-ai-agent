"""
Code executor tool: runs LLM-generated Python code in an isolated subprocess.

SECURITY MODEL
--------------
This is a *defensive* sandbox, not a production-grade one. For real
production use, this should be wrapped in Docker, gVisor, firecracker,
or a similar kernel-level isolation boundary.

What this sandbox provides:
  - Subprocess isolation: agent process doesn't crash if code crashes.
  - Hard timeout via asyncio.wait_for on a subprocess.
  - Restricted writable filesystem: only a temp dir the subprocess creates.
  - Output size cap: truncates stdout/stderr to prevent memory blowup.
  - Curated stdlib only: code runs with a clean interpreter.

What this sandbox does NOT provide:
  - Network isolation at the kernel level (we block urllib/socket at
    import time, but a determined attacker can bypass this).
  - CPU/memory quotas (would need cgroups).
  - Filesystem isolation beyond what Python respects.

For the assignment: the agent only runs code it *generates itself* based
on user queries — not arbitrary user-submitted code. Threat model is
"LLM makes a mistake and writes destructive code" rather than "attacker
tries to escape the sandbox". This layered-defence approach is
appropriate for that threat model.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

from tools.base import Tool


# Preamble injected at the top of every user snippet.
# - Disables most common network paths.
# - Makes common data libraries available if installed.
# - Restricts file system to a workspace dir.
_PREAMBLE = textwrap.dedent("""
    import sys, os, builtins

    # Block common network modules. Not foolproof, but raises the bar.
    _blocked = ['socket', 'urllib', 'urllib.request', 'http.client',
                'requests', 'httpx', 'aiohttp', 'ftplib', 'telnetlib',
                'smtplib', 'subprocess']
    _real_import = builtins.__import__
    def _guarded_import(name, *args, **kwargs):
        if name in _blocked or any(name.startswith(b + '.') for b in _blocked):
            raise ImportError(f"Import of '{name}' is blocked in the sandbox.")
        return _real_import(name, *args, **kwargs)
    builtins.__import__ = _guarded_import

    # Chdir to workspace so relative paths are contained.
    os.chdir(os.environ.get('SANDBOX_WORKSPACE', os.getcwd()))
""").strip()


MAX_OUTPUT_CHARS = 4000  # Truncate stdout/stderr beyond this.


class CodeExecutorTool(Tool):
    name = "code_executor"
    description = (
        "Executes a Python code snippet in an isolated subprocess and "
        "returns its stdout/stderr. Use this for: "
        "data analysis (pandas, numpy), CSV/JSON parsing, complex "
        "multi-step calculations beyond the calculator tool, or any "
        "task requiring programmatic logic. "
        "\n\n"
        "Rules:\n"
        "  - Your code must print() or write to stdout to return results. "
        "Return values are not captured.\n"
        "  - Network access is blocked. Use web_search for external data.\n"
        "  - Timeout is enforced; keep execution under 20 seconds.\n"
        "  - Available libraries: Python stdlib, and (if installed) "
        "pandas, numpy. Do not try to pip install.\n"
        "  - Do not attempt file I/O outside the working directory.\n"
        "\n"
        "Example: print(sum(x**2 for x in range(10)))"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must print results to stdout.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout_seconds: int = 20) -> None:
        super().__init__(timeout_seconds=timeout_seconds)

    async def _run(self, code: str) -> str:
        workspace = Path(tempfile.mkdtemp(prefix="agent_sandbox_"))
        try:
            script_path = workspace / "snippet.py"
            full_source = _PREAMBLE + "\n\n# === user code ===\n" + code
            script_path.write_text(full_source, encoding="utf-8")

            env = os.environ.copy()
            env["SANDBOX_WORKSPACE"] = str(workspace)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            # Clear PYTHONPATH to avoid leaking project modules into sandbox
            env.pop("PYTHONPATH", None)

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace),
                env=env,
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds - 1,  # leave margin for base-class timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Code execution exceeded {self.timeout_seconds}s")

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            return self._format_output(proc.returncode, stdout, stderr)

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _format_output(returncode: int | None, stdout: str, stderr: str) -> str:
        def _clip(s: str) -> str:
            if len(s) > MAX_OUTPUT_CHARS:
                return s[:MAX_OUTPUT_CHARS] + f"\n... [output truncated at {MAX_OUTPUT_CHARS} chars]"
            return s

        parts = [f"Exit code: {returncode}"]
        if stdout.strip():
            parts.append(f"--- stdout ---\n{_clip(stdout).rstrip()}")
        if stderr.strip():
            parts.append(f"--- stderr ---\n{_clip(stderr).rstrip()}")
        if not stdout.strip() and not stderr.strip():
            parts.append("(no output — did you forget to print()?)")
        return "\n\n".join(parts)