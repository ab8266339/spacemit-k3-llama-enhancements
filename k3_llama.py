#!/usr/bin/env python3
"""Start or compare the two SpacemiT llama.cpp execution paths on K3."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, ProxyHandler, Request
from uuid import uuid4

PROFILES = {"legacy64": (False, 64), **{f"standalone{n}": (True, n) for n in (64, 128, 256, 512)}}
DEFAULT_PROFILES = ("legacy64", "standalone64", "standalone128")
LABELS = ("春", "夏", "秋", "冬")
BODY = "这是一项端侧推理实验。语言模型逐个生成词元，专家路由根据当前输入选择参与计算的专家。测试记录文本处理吞吐量和逐词生成速度。"
SUFFIX = "请用中文说明为什么输入处理和逐词生成应当分别计时。"
LOCAL_HTTP = build_opener(ProxyHandler({}))


def configuration(server, model, profile, port, lib_dirs):
    standalone, batch = PROFILES[profile]
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    dirs = [str(server.parent.parent / "lib"), *(str(d) for d in lib_dirs), *existing]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(d for d in dirs if d))
    env.update(GGML_SPACEMIT_WORKERS="8", SPACEMIT_PERFER_CORE_ID="8,9,10,11,12,13,14,15", SPACEMIT_PERFER_CORE_ARCH="a064",
               GGML_SPACEMIT_BACKEND="1" if standalone else "0")
    device = ["--device", "SPACEMIT0", "-ngl", "99"] if standalone else ["--device", "none", "-ngl", "0", "--no-op-offload"]
    args = [str(server), "-m", str(model), "-t", "8", "-tb", "8", "-np", "1", "-c", "1024",
            "-b", str(batch), "-ub", str(batch), "--reasoning", "off", "--no-warmup",
            "--host", "127.0.0.1", "--port", str(port), "-lv", "4", *device]
    return args, env


def api(port, path, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(f"http://127.0.0.1:{port}{path}", data=body, headers={"Content-Type": "application/json"})
    with LOCAL_HTTP.open(req, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def session(server, model, profile, lib_dirs, logfile):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    alias = "k3-bench-" + uuid4().hex
    args, env = configuration(server, model, profile, port, lib_dirs)
    start = time.monotonic()
    with logfile.open("xb") as log:
        proc = subprocess.Popen([*args, "--alias", alias], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.monotonic() - start < 600:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited with status {proc.returncode}. See {logfile}")
                try:
                    if api(port, "/health", timeout=2).get("status") == "ok":
                        ids = [m["id"] for m in api(port, "/v1/models")["data"]]
                        if alias not in ids:
                            raise RuntimeError("A different server answered on the benchmark port")
                        break
                except HTTPError as e:
                    if e.code != 503:
                        raise
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Model loading timed out. See {logfile}")
            yield port, time.monotonic() - start, args
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def fixed_prompt(prefix, body, suffix, count=128):
    room = count - len(prefix) - len(suffix)
    if room < 0 or len(body) < room:
        raise ValueError(f"Tokenizer cannot form the fixed {count}-token prompt")
    return [*prefix, *body[:room], *suffix]


def check_round(timing, count=128):
    if (timing.get("prompt_n"), timing.get("predicted_n"), timing.get("cache_n")) != (count, 64, 0):
        raise ValueError(f"Expected {count} uncached input tokens and 64 output tokens. Received {timing}")


def benchmark(port, count):
    def tokens(text):
        return api(port, "/tokenize", {"content": text, "add_special": False})["tokens"]
    body, suffix = tokens(BODY * 22), tokens(SUFFIX)
    rounds = []
    for index, label in enumerate(LABELS):
        ids = fixed_prompt(tokens(label), body, suffix, count)
        start = time.monotonic()
        response = api(port, "/completion", {"prompt": ids, "n_predict": 64, "cache_prompt": False,
                       "ignore_eos": True, "temperature": 0, "seed": 17, "stream": False})
        elapsed = time.monotonic() - start
        timing = response["timings"]
        check_round(timing, count)
        rounds.append({"label": label, "warmup": index == 0, "content": response["content"],
                       "prompt_sha256": hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest(), "wall_seconds": elapsed, "timings": timing})
    medians = {m: statistics.median(r["timings"][m] for r in rounds[1:])
               for m in ("prompt_per_second", "predicted_per_second")}
    return {"rounds": rounds, "median_tokens_per_second": medians,
            "median_request_seconds": statistics.median(r["wall_seconds"] for r in rounds[1:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server", default="llama-server", help="SpacemiT llama-server executable")
    common.add_argument("--model", required=True, type=Path, help="Local GGUF weights")
    common.add_argument("--lib-dir", action="append", type=Path, default=[], help="Additional runtime library directory (repeatable)")
    modes = parser.add_subparsers(dest="command", required=True)
    serve = modes.add_parser("serve", parents=[common])
    serve.add_argument("--profile", choices=PROFILES, default="standalone128")
    serve.add_argument("--port", type=int, default=8080)
    bench = modes.add_parser("bench", parents=[common])
    bench.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(DEFAULT_PROFILES))
    bench.add_argument("--prompt-tokens", type=int, choices=(128, 512), default=128,
                       help="Uncached input tokens per request; each request generates 64 tokens")
    bench.add_argument("--out", required=True, type=Path, help="Parent directory for a new run")
    opts = parser.parse_args()
    location = shutil.which(opts.server)
    if not location:
        parser.error(f"Cannot locate executable {opts.server}")
    server = Path(location).resolve()
    model = opts.model.resolve(strict=True)
    if opts.command == "serve":
        args, env = configuration(server, model, opts.profile, opts.port, opts.lib_dir)
        os.execve(server, args, env)
    # Unwind the active session on a normal service-manager termination.
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    if len(set(opts.profiles)) != len(opts.profiles):
        parser.error("Profiles in one run must be unique")
    opts.out.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="k3-run-", dir=opts.out))
    print(out, flush=True)
    previous = None
    latencies = {}
    for profile in opts.profiles:
        with session(server, model, profile, opts.lib_dir, out / f"{profile}.log") as (port, load, args):
            result = benchmark(port, opts.prompt_tokens)
        hashes = [r["prompt_sha256"] for r in result["rounds"]]
        if previous is not None and previous != hashes:
            raise ValueError("The profiles received different token sequences")
        previous = hashes
        result.update(profile=profile, load_seconds=load, arguments=args[1:], model_bytes=model.stat().st_size)
        (out / f"{profile}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(profile, result["median_tokens_per_second"], flush=True)
        latencies[profile] = result["median_request_seconds"]
    fastest = min(latencies, key=latencies.get)
    ranking = {"prompt_tokens": opts.prompt_tokens, "generated_tokens": 64, "fastest_measured_profile": fastest,
               "median_request_seconds": latencies}
    (out / "ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(f"Lowest measured median request latency: {fastest} ({latencies[fastest]:.3f} s)", flush=True)


if __name__ == "__main__":
    main()
