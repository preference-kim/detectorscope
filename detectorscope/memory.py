"""macOS process footprint accounting and a separate worker watchdog.

phys_footprint includes CPU and Metal memory, including compressed/swapped
pages. RSS and MPS allocator counters must not be added to it. The watchdog
terminates a worker after a breach; it cannot prevent transient overshoot.
"""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import select
import signal
import subprocess
import sys


GIB = 2**30
POLL_SECONDS = 0.05


class _RUsageInfoV4(ctypes.Structure):
  # Stable RUSAGE_INFO_V4 ABI from macOS <sys/resource.h>. Request this exact
  # version rather than CURRENT, which can outgrow the supplied buffer.
  _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
      (name, ctypes.c_uint64) for name in """
      ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups
      ri_pageins ri_wired_size ri_resident_size ri_phys_footprint
      ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time
      ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups
      ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread
      ri_diskio_byteswritten ri_cpu_time_qos_default ri_cpu_time_qos_maintenance
      ri_cpu_time_qos_background ri_cpu_time_qos_utility ri_cpu_time_qos_legacy
      ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive
      ri_billed_system_time ri_serviced_system_time ri_logical_writes
      ri_lifetime_max_phys_footprint ri_instructions ri_cycles ri_billed_energy
      ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time
      """.split()]


@dataclass(frozen=True)
class ProcessMemory:
  footprint_bytes: int
  peak_footprint_bytes: int
  start_time: int


@lru_cache(maxsize=1)
def _libproc():
  if sys.platform != "darwin":
    raise RuntimeError("Process footprint budgeting requires macOS.")
  library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
  library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
  library.proc_pid_rusage.restype = ctypes.c_int
  return library


def process_memory(pid=None):
  """Read current and lifetime peak footprint, without loading PyTorch."""
  usage = _RUsageInfoV4()
  if _libproc().proc_pid_rusage(
      os.getpid() if pid is None else pid, 4, ctypes.byref(usage)):
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
  return ProcessMemory(usage.ri_phys_footprint,
                       usage.ri_lifetime_max_phys_footprint,
                       usage.ri_proc_start_abstime)


class MemoryWatchdog:
  """Watch this process from a small stdlib-only child (0 disables it).

  Use around the worker's entire lifetime, before importing the model runtime.
  A separate process can still sample while a native call holds the worker GIL.
  The budget covers this worker; the watchdog's own memory is separate.
  """

  def __init__(self, limit_gib):
    if not math.isfinite(limit_gib * GIB) or limit_gib < 0:
      raise ValueError("memory limit must be finite and nonnegative")
    self.limit_bytes = math.ceil(limit_gib * GIB)
    self.process = None

  def __enter__(self):
    if not self.limit_bytes:
      return self
    usage = process_memory()
    if usage.peak_footprint_bytes > self.limit_bytes:
      raise MemoryError("Worker already exceeds the requested memory budget.")
    self.process = subprocess.Popen(
        [sys.executable, "-m", "detectorscope.memory", str(os.getpid()),
         str(usage.start_time), str(self.limit_bytes)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
    )
    try:
      ready, _, _ = select.select([self.process.stdout], [], [], 10)
      if not ready or self.process.stdout.readline() != b"ready\n":
        raise RuntimeError("Memory watchdog failed to start.")
    except BaseException:
      self.__exit__(None, None, None)
      raise
    finally:
      self.process.stdout.close()
    return self

  def __exit__(self, *exc):
    if self.process is not None:
      # EOF also stops the child if the worker crashes or is killed.
      self.process.stdin.close()
      try:
        self.process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait()

  def status(self):
    if not self.limit_bytes:
      return {"enabled": False}
    if self.process is None or self.process.poll() is not None:
      raise RuntimeError("Memory watchdog is not running.")
    usage = process_memory()
    return {"enabled": True, "metric": "phys_footprint",
            "limit_bytes": self.limit_bytes,
            "footprint_bytes": usage.footprint_bytes,
            "peak_footprint_bytes": usage.peak_footprint_bytes,
            "enforcement": "terminate_worker_after_breach",
            "poll_interval_seconds": POLL_SECONDS}


def _watch_parent(pid, start_time, limit_bytes):
  # Never signal an arbitrary PID or a new process that reused the worker PID.
  if pid != os.getppid():
    raise ValueError("The watchdog may only monitor its own parent.")
  announced = False
  while os.getppid() == pid:
    ready, _, _ = select.select([sys.stdin.buffer], [], [], 0)
    if ready and not os.read(sys.stdin.fileno(), 1):
      return
    try:
      usage = process_memory(pid)
      if usage.start_time != start_time:
        return
      # The kernel's lifetime peak catches brief spikes between samples too.
      if usage.peak_footprint_bytes > limit_bytes:
        reason = (f"Memory budget exceeded: peak footprint "
                  f"{usage.peak_footprint_bytes / GIB:.3f} GiB > "
                  f"{limit_bytes / GIB:.3f} GiB")
      else:
        reason = None
    except OSError as error:
      reason = f"Cannot monitor worker memory: {error}"
    if reason:
      print(f"{reason}; terminating worker PID {pid}.", file=sys.stderr,
            flush=True)
      if os.getppid() == pid:
        os.kill(pid, signal.SIGTERM)
      return
    if not announced:
      print("ready", flush=True)
      announced = True
    # The pipe wakes us immediately on normal shutdown; no polling of a dead
    # or reparented process, and no dependence on the worker's Python threads.
    select.select([sys.stdin.buffer], [], [], POLL_SECONDS)


if __name__ == "__main__":
  _watch_parent(*(int(arg) for arg in sys.argv[1:]))
