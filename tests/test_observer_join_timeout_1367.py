"""Exercise the actual shutdown helper without launching a host observer."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_join_timeout_stays_failed_after_late_sampler_retirement():
    source = Path(__file__).resolve().parents[1] / "tools/pq_row_profile_observer.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "finish_netdata")
    states = iter([True, False])
    joined, collected, errors, events = [], [], [], []
    sampler = SimpleNamespace(
        stop=lambda timeout: joined.append(timeout),
        is_alive=lambda: next(states))
    space = {"nd": sampler, "collect_netdata": lambda: collected.append(True),
             "telemetry_errors": errors,
             "event": lambda *args, **kwargs: events.append((args, kwargs))}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[helper], type_ignores=[])), str(source), "exec"), space)
    space["finish_netdata"]()
    assert joined == [90]
    assert not collected, "final collection raced a still-live periodic tick"
    assert not sampler.is_alive()  # retires while the caller cleans up py-spy
    assert errors, "late retirement erased the missing final telemetry window"
