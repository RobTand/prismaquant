"""Only owner bookkeeping remains; asynchronous IO belongs to io_engine."""
import threading
from prismaquant.produced_stager import OwnerLock


def test_the_owner_lock_is_given_up_around_a_call_and_taken_back():
    lock, seen = OwnerLock(), []
    attempted, acquired = threading.Event(), threading.Event()
    def other():
        attempted.set()
        with lock.held():
            seen.append('other')
            acquired.set()
    with lock.held():
        with lock.held():
            worker = threading.Thread(target=other)
            worker.start()
            assert attempted.wait(30)
            assert lock._owner == threading.get_ident()
            with lock.yielded():
                assert acquired.wait(30)
                worker.join(30)
                assert not worker.is_alive()
        assert lock._owner == threading.get_ident() and seen == ['other']
    assert lock._owner is None
    with lock.yielded():
        with lock.held():
            assert lock._owner == threading.get_ident()
