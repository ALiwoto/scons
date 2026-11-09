# wgodot-changes::file

"""Opt-in build timings. SCONS_DIAGNOSTICS names the output directory.

Operations aggregate across threads; nested self times exclude measured children.
Command intervals measure SPAWN calls, including waiting, not subprocess CPU time.
JSONL remains readable during a build and after interruption. Memory use depends
on operation names and thread nesting, not the number of build targets.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TypedDict, TypeVar, cast


@dataclass
class _Frame:
    wall: int
    cpu: int
    child_wall: int = 0
    child_cpu: int = 0


@dataclass
class _Timing:
    calls: int = 0
    wall: int = 0
    cpu: int = 0
    self_wall: int = 0
    self_cpu: int = 0
    maximum: int = 0

    def report(self) -> dict[str, int | float]:
        return {
            'calls': self.calls,
            'wall_seconds': self.wall / 1e9,
            'thread_cpu_seconds': self.cpu / 1e9,
            'self_wall_seconds': self.self_wall / 1e9,
            'self_thread_cpu_seconds': self.self_cpu / 1e9,
            'max_seconds': self.maximum / 1e9,
        }


class _Activity(TypedDict):
    id: int
    kind: str
    targets: list[str]
    sources: list[str]
    executable: str
    started_ns: int


class Recorder:
    def __init__(self, directory: str) -> None:
        output = Path(directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path = output / f'scons-{stamp}-{os.getpid()}.jsonl'
        self._output = self.path.open('w', encoding='utf-8', buffering=1)
        self._text = self.path.with_suffix('.log').open(
            'w', encoding='utf-8', buffering=1
        )
        self._lock = threading.Lock()
        self._local = threading.local()
        self._start = time.perf_counter_ns()
        self._utc_start = time.time_ns()
        self._start_cpu = time.process_time_ns()
        self._last_change = self._start
        self._active = 0
        self._maximum = 0
        self._busy = 0
        self._slots = 0
        self._started = 0
        self._finished = 0
        self._next_id = 0
        self._running: dict[int, _Activity] = {}
        self._stop = threading.Event()
        self.exit_code: int | None = None
        self._timings: dict[str, _Timing] = {}
        self._event('start', self._start, pid=os.getpid(), cwd=os.getcwd())
        atexit.register(self.finish)
        self._heartbeat = threading.Thread(
            target=self._progress, name='SCons diagnostics', daemon=True
        )
        self._heartbeat.start()
        print(f'SCons diagnostics: {self.path}', flush=True)

    def _event(self, kind: str, now: int, **fields: object) -> None:
        record: dict[str, object] = {
            'event': kind,
            'utc': datetime.fromtimestamp(
                (self._utc_start + now - self._start) / 1e9, timezone.utc
            ).isoformat(timespec='milliseconds'),
            'elapsed_seconds': (now - self._start) / 1e9,
        }
        record.update(fields)
        self._output.write(json.dumps(record, separators=(',', ':')) + '\n')
        if kind in ('command_start', 'command_end', 'function_start', 'function_end'):
            details = (
                f"{kind.upper()} #{fields['id']} {fields['executable']} "
                f"source={json.dumps(fields['sources'])} target={json.dumps(fields['targets'])}"
            )
            if kind.endswith('_end'):
                details += (
                    f" seconds={fields['seconds']:.6f} exit={fields['exit_code']}"
                )
            self._message(record, details)
        elif kind == 'snapshot':
            self._message(
                record, 'ACTIVE ' + json.dumps(fields['running'], separators=(',', ':'))
            )
        elif kind == 'finish':
            self._message(
                record,
                f"FINISH exit={fields['exit_code']} summary={self.path.with_suffix('.json')}",
            )

    def _message(self, record: dict[str, object], details: str) -> None:
        line = f"[{record['utc']} +{record['elapsed_seconds']:.6f}s pid={os.getpid()}] {details}"
        self._text.write(line + '\n')
        print(line, flush=True)

    def _progress(self) -> None:
        while not self._stop.wait(5):
            with self._lock:
                now = time.perf_counter_ns()
                self._event('snapshot', now, **self._report(now))

    def note(self, name: str, **fields: object) -> None:
        with self._lock:
            self._event('note', time.perf_counter_ns(), name=name, **fields)

    def _command_clock(self, now: int) -> None:
        elapsed = now - self._last_change
        if self._active:
            self._busy += elapsed
        self._slots += elapsed * self._active
        self._last_change = now

    def _report(self, now: int) -> dict[str, object]:
        self._command_clock(now)
        return {
            'exit_code': self.exit_code,
            'elapsed_seconds': (now - self._start) / 1e9,
            'scons_process_cpu_seconds': (time.process_time_ns() - self._start_cpu)
            / 1e9,
            'spawn_commands': {
                'started': self._started,
                'finished': self._finished,
                'active': self._active,
                'max_parallel': self._maximum,
                'busy_wall_seconds': self._busy / 1e9,
                'summed_wall_seconds': self._slots / 1e9,
            },
            'no_spawn_command_seconds': (now - self._start - self._busy) / 1e9,
            'running': [
                dict(activity, seconds=(now - activity['started_ns']) / 1e9)
                for activity in self._running.values()
            ],
            'operations': {
                name: timing.report() for name, timing in self._timings.items()
            },
        }

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        previous: _Frame | None = getattr(self._local, 'frame', None)
        frame = _Frame(time.perf_counter_ns(), time.thread_time_ns())
        self._local.frame = frame
        try:
            yield
        finally:
            now = time.perf_counter_ns()
            wall = now - frame.wall
            cpu = time.thread_time_ns() - frame.cpu
            self._local.frame = previous
            if previous is not None:
                previous.child_wall += wall
                previous.child_cpu += cpu
            with self._lock:
                timing = self._timings.get(name)
                if timing is None:
                    timing = _Timing()
                    self._timings[name] = timing
                timing.calls += 1
                timing.wall += wall
                timing.cpu += cpu
                timing.self_wall += wall - frame.child_wall
                timing.self_cpu += cpu - frame.child_cpu
                timing.maximum = max(timing.maximum, wall)
                if wall >= 50_000_000:
                    self._event(
                        'operation',
                        now,
                        name=name,
                        seconds=wall / 1e9,
                        thread=threading.get_ident(),
                    )

    def run(
        self,
        kind: str,
        targets: list[str],
        sources: list[str],
        executable: str,
        arguments: list[str],
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        with self._lock:
            start = time.perf_counter_ns()
            if kind == 'command':
                self._command_clock(start)
                self._active += 1
                self._maximum = max(self._maximum, self._active)
                self._started += 1
            self._next_id += 1
            command_id = self._next_id
            self._running[command_id] = {
                'id': command_id,
                'kind': kind,
                'targets': targets,
                'sources': sources,
                'executable': executable,
                'started_ns': start,
            }
            self._event(
                kind + '_start',
                start,
                id=command_id,
                targets=targets,
                sources=sources,
                executable=executable,
                arguments=arguments,
                active=self._active,
            )
        exit_code: int | None = None
        try:
            with self.span(
                'external_command' if kind == 'command' else 'python_action'
            ):
                result = function(*args, **kwargs)
            exit_code = 0 if not result else result if isinstance(result, int) else 1
            return result
        finally:
            with self._lock:
                end = time.perf_counter_ns()
                if kind == 'command':
                    self._command_clock(end)
                    self._active -= 1
                    self._finished += 1
                del self._running[command_id]
                self._event(
                    kind + '_end',
                    end,
                    id=command_id,
                    targets=targets,
                    sources=sources,
                    executable=executable,
                    seconds=(end - start) / 1e9,
                    exit_code=exit_code,
                    active=self._active,
                )

    def finish(self) -> None:
        self._stop.set()
        self._heartbeat.join()
        with self._lock:
            if self._output.closed:
                return
            now = time.perf_counter_ns()
            report = self._report(now)
            self._event('finish', now, **report)
            self.path.with_suffix('.json').write_text(
                json.dumps(report, indent=2), encoding='utf-8'
            )
            self._output.close()
            self._text.close()


_directory = os.environ.get('SCONS_DIAGNOSTICS')
recorder = Recorder(_directory) if _directory else None
_Function = TypeVar('_Function', bound=Callable[..., object])


def measure(function: _Function) -> _Function:
    """Disabled diagnostics leave the original callable untouched."""
    active = recorder
    if active is None:
        return function
    name = function.__module__ + '.' + function.__qualname__

    @wraps(function)
    def measured(*args: object, **kwargs: object) -> object:
        with active.span(name):
            return function(*args, **kwargs)

    return cast(_Function, measured)
