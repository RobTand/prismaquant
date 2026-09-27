"""Import spelling must not hide a new loader from the #1294 freeze."""
import pytest

from test_io_site_freeze import _sites


@pytest.mark.parametrize("source,expected", [
    ("from concurrent.futures import ThreadPoolExecutor as Readers\n"
     "def load():\n    Readers(2)\n", ["load"]),
    ("from concurrent.futures import ProcessPoolExecutor as Workers\n"
     "Workers()\n", ["<module>"]),
    ("from threading import Thread as Background\n"
     "Background(target=print)\n", ["<module>"]),
    ("from multiprocessing import Pool as Workers, Process as Child\n"
     "Workers(2)\nChild(target=print)\n", ["<module>", "<module>"]),
    ("import multiprocessing as backend\nbackend.Pool(2)\n", ["<module>"]),
    ("import multiprocessing as backend\n"
     "backend.get_context('spawn').Process(target=print)\n", ["<module>"]),
    ("from threading import Thread as Background\n"
     "class Reader(Background):\n    pass\n", ["Reader (subclass)"]),
    ("from concurrent.futures import ThreadPoolExecutor as Readers\n"
     "class NewLoader(Readers):\n    pass\n", ["NewLoader (subclass)"]),
    ("from concurrent.futures import ThreadPoolExecutor\n"
     "Readers = ThreadPoolExecutor\nReaders()\n", ["<module>"]),
    ("def load():\n    Readers()\n"
     "from concurrent.futures import ThreadPoolExecutor as Readers\n", ["load"]),
])
def test_aliases_do_not_hide_sites(source, expected):
    assert _sites(source) == expected


def test_function_local_alias_does_not_leak_into_another_function():
    source = (
        "def pooled():\n"
        "    from concurrent.futures import ThreadPoolExecutor as Factory\n"
        "    Factory()\n"
        "def ordinary():\n"
        "    from user_api import Factory\n"
        "    Factory()\n"
    )
    assert _sites(source) == ["pooled"]


def test_later_rebinding_does_not_hide_an_earlier_pool():
    source = (
        "from concurrent.futures import ThreadPoolExecutor as Factory\n"
        "Factory()\n"
        "from user_api import Factory\n"
    )
    assert _sites(source) == ["<module>"]
