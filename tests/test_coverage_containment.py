"""Coverage containment on a database of three tests with known covered lines (PQ #2806)."""
import gzip
import io
import sys
from pathlib import Path

import coverage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import coverage_containment


def test_three_tests_give_known_ranges_and_containment(tmp_path):
    data = coverage.CoverageData(basename=str(tmp_path / '.coverage'))
    for context, lines in [('', {'a.py': [1, 2, 3, 4], 'b.py': [9]}),
                           ('t.py::small|run', {'a.py': [1, 3]}),
                           ('t.py::big|setup', {'a.py': [1]}),
                           ('t.py::big|run', {'a.py': [2, 3], 'b.py': [5, 7]})]:
        data.set_context(context)
        data.add_lines(lines)
    data.write()
    (tmp_path / 'outcomes.tsv').write_text('t.py::small\tpassed\nt.py::big\tpassed\nt.py::child\tpassed\n')
    ranges = tmp_path / 'ranges.tsv.gz'
    coverage_containment.build_line_ranges(tmp_path / '.coverage', ranges, tmp_path / 'outcomes.tsv')
    assert ranges.read_bytes()[3:8] == bytes(5)  # gzip header: no file name, time 0
    assert gzip.open(ranges, 'rt').read() == (
        't.py::big\ta.py\t1-3\nt.py::big\tb.py\t5,7\nt.py::child\tNOT MEASURED\tpassed\nt.py::small\ta.py\t1,3\n')
    answer = io.StringIO()
    coverage_containment.query(ranges, ['t.py', 't.py::gone'], show_lines=True, out=answer)
    assert answer.getvalue() == (
        'not-measured-count\t1\n'
        'lines\tt.py::big\t5\nrange\tt.py::big\ta.py\t1-3\nrange\tt.py::big\tb.py\t5,7\n'
        'not-measured\tt.py::child\tpassed\n'
        'lines\tt.py::small\t2\nrange\tt.py::small\ta.py\t1,3\ncontained-by\tt.py::small\tt.py::big\t5\n'
        'unknown\tt.py::gone\n')
