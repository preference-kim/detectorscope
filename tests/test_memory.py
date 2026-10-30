"""Test the watchdog in disposable processes; never cap the test runner."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest

from detectorscope.memory import MemoryWatchdog, _watch_parent


ROOT = Path(__file__).resolve().parents[1]
MAC_ONLY = pytest.mark.skipif(sys.platform != "darwin", reason="macOS footprint API")


def run_python(source):
  return subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                        cwd=ROOT, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf"), 1e308])
def test_invalid_budget_cannot_silently_disable_monitoring(limit):
  with pytest.raises(ValueError):
    MemoryWatchdog(limit)


def test_zero_disables_and_positive_sub_byte_budget_does_not():
  with MemoryWatchdog(0) as guard:
    assert guard.process is None
    assert guard.status() == {"enabled": False}
  assert MemoryWatchdog(1e-30).limit_bytes == 1


def test_watchdog_refuses_to_signal_an_unrelated_process():
  with pytest.raises(ValueError, match="own parent"):
    _watch_parent(os.getpid(), 0, 1)


@MAC_ONLY
def test_native_footprint_tracks_touched_cpu_memory_and_clean_shutdown():
  result = run_python('''
    import json
    from detectorscope.memory import MemoryWatchdog, process_memory

    with MemoryWatchdog(0.5) as guard:
        before = process_memory()
        values = bytearray(32 * 2**20)
        for i in range(0, len(values), 4096):
            values[i] = 1
        after = process_memory()
        status = guard.status()
        assert after.footprint_bytes > before.footprint_bytes + 16 * 2**20
        assert after.peak_footprint_bytes >= after.footprint_bytes
        assert before.start_time == after.start_time
        assert status['limit_bytes'] == 2**29
        assert status['metric'] == 'phys_footprint'
    print(json.dumps({'exit_code':guard.process.returncode, 'status':status}))
  ''')
  assert result.returncode == 0, result.stderr
  assert json.loads(result.stdout)["exit_code"] == 0


@MAC_ONLY
@pytest.mark.parametrize("after_allocation", [
    # PyDLL holds the worker GIL throughout the native sleep.
    "ctypes.PyDLL(None).sleep(5)",
    # The lifetime peak must catch a spike even if it has already been freed.
    "del values; time.sleep(5)",
])
def test_exceeding_budget_terminates_worker_even_during_native_call(after_allocation):
  result = run_python(f'''
    import ctypes
    import time
    from detectorscope.memory import MemoryWatchdog

    with MemoryWatchdog(0.0625):
        values = bytearray(96 * 2**20)
        for i in range(0, len(values), 4096):
            values[i] = 1
        {after_allocation}
        raise AssertionError('Over-budget worker kept running')
  ''')
  assert result.returncode == -signal.SIGTERM, result.stderr
  assert "Memory budget exceeded" in result.stderr
  assert "terminating worker PID" in result.stderr


@MAC_ONLY
def test_budget_cannot_start_above_the_existing_peak():
  result = run_python('''
    from detectorscope.memory import MemoryWatchdog
    try:
        with MemoryWatchdog(1e-6):
            raise AssertionError('Undersized budget was accepted')
    except MemoryError as error:
        assert 'already exceeds' in str(error)
  ''')
  assert result.returncode == 0, result.stderr
