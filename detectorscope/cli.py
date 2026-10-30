"""Local CLI and loopback HTTP worker for document agents."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  serve = sub.add_parser("serve", help="Run a bounded layer-streaming worker")
  serve.add_argument("--port", type=int, default=8766)
  serve.add_argument("--model", help="Default: liamdugan/detector-scope")
  serve.add_argument("--revision", help="Default: the pinned model revision")
  serve.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                     default="auto")
  serve.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                     default="float32")
  serve.add_argument("--local-files-only", action="store_true")
  serve.add_argument("--max-tokens", type=int, default=1024)
  serve.add_argument("--batch-size", type=int, default=1,
                     help="Maximum chunks per forward; weights are shared across chunks")
  serve.add_argument("--cache-size", type=int, default=128)
  serve.add_argument("--memory-limit-gib", type=float,
                     default=3.0 if sys.platform == "darwin" else 0.0,
                     help="macOS whole-worker footprint budget in GiB; "
                          "terminates the worker after a breach "
                          "(default: 3 on macOS; 0 disables)")
  check = sub.add_parser("check", help="Check UTF-8 text via the running worker")
  check.add_argument("file", help="UTF-8 text/Markdown file, or - for stdin")
  check.add_argument("--port", type=int, default=8766)
  check.add_argument("--highlights", action="store_true")
  check.add_argument("--top-features", type=int, default=10)
  check.add_argument("--timeout", type=float, default=300)
  args = parser.parse_args()
  if args.command == "check":
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    body = json.dumps({"text": text, "highlights": args.highlights,
                       "top_features": args.top_features}).encode()
    request = Request(f"http://127.0.0.1:{args.port}/analyze", data=body,
                      headers={"Content-Type": "application/json"})
    try:
      with urlopen(request, timeout=args.timeout) as response:
        print(response.read().decode())
    except HTTPError as e:
      parser.exit(1, f"Worker rejected the request: {e.read().decode()}\n")
    except (URLError, TimeoutError) as e:
      parser.exit(1, f"Worker unavailable: {e}. Start detectorscope serve.\n")
    return

  if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib < 0:
    parser.error("--memory-limit-gib must be finite and nonnegative")
  if args.memory_limit_gib and sys.platform != "darwin":
    parser.error("--memory-limit-gib requires macOS")
  from detectorscope.memory import MemoryWatchdog

  # Start monitoring before imports, checkpoint loading and warmup as well as
  # requests, cached results and idle time. Keep the existing MPS budget too.
  with MemoryWatchdog(args.memory_limit_gib) as memory:
    _serve(args, memory)


def _serve(args, memory):
  from detectorscope.documents import DocumentAnalyzer
  from detectorscope.inference import DEFAULT_MODEL, DEFAULT_REVISION, OptimizedDetector

  started = time.perf_counter()
  detector = OptimizedDetector.from_pretrained(
      args.model or DEFAULT_MODEL, revision=args.revision or DEFAULT_REVISION,
      device=args.device,
      torch_dtype=args.dtype, local_files_only=args.local_files_only,
  )
  analyzer = DocumentAnalyzer(detector, args.max_tokens, args.cache_size,
                              args.batch_size)
  # Pay kernel initialization once, before announcing readiness.
  detector("A short document used to initialize the local detector.")

  class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload):
      body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
      self.send_response(status)
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      if self.path != "/health":
        self.reply(404, {"error": "Not found"})
        return
      try:
        memory_status = memory.status()
      except (OSError, RuntimeError) as error:
        self.reply(503, {"ready": False, "error": str(error)})
        return
      self.reply(200, {"ready": True, "model": detector.identity,
                       "max_batch_size": analyzer.batch_size,
                       "max_tokens": analyzer.max_tokens,
                       "memory": memory_status})

    def do_POST(self):
      if self.path != "/analyze":
        self.reply(404, {"error": "Not found"})
        return
      try:
        memory.status()  # Refuse inference if the watchdog unexpectedly died.
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 2 * 1024 * 1024:
          self.reply(413, {"error": "Request must be at most 2 MiB."})
          return
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
          raise ValueError("Request must be a JSON object.")
        start = time.perf_counter()
        result = analyzer.analyze(data.get("text"), data.get("highlights", False),
                                  data.get("top_features", 10))
        result["elapsed_seconds"] = time.perf_counter() - start
        self.reply(200, result)
      except (ValueError, TypeError) as error:
        self.reply(400, {"error": str(error)})
      except Exception as error:
        self.log_error("Inference failed: %r", error)
        self.reply(500, {"error": "Inference failed; see worker log."})

  server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
  print(json.dumps({"ready": True, "port": args.port,
                    "startup_seconds": time.perf_counter() - started,
                    "model": detector.identity,
                    "memory": memory.status()}), flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
