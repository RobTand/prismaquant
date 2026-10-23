import threading

import pytest

from prismaquant import cost_stage_checkpoint as checkpoint


@pytest.mark.parametrize('nofollow', [False, True])
def test_same_process_publishers_keep_their_own_bytes(tmp_path, monkeypatch, nofollow):
    destination = tmp_path / 'published'
    second_ready = threading.Event()
    first_linked = threading.Event()
    real_link = checkpoint.os.link
    results = {}
    errors = []

    def ordered_link(*args, **kwargs):
        if threading.current_thread().name == 'publisher-a':
            assert second_ready.wait(10)
            try:
                return real_link(*args, **kwargs)
            finally:
                first_linked.set()
        second_ready.set()
        assert first_linked.wait(10)
        return real_link(*args, **kwargs)

    def publish(name, payload):
        try:
            results[name] = checkpoint.publish_new_bytes(destination, payload, nofollow=nofollow)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(checkpoint.os, 'link', ordered_link)
    first = threading.Thread(target=publish, args=('a', b'first payload'), name='publisher-a')
    second = threading.Thread(target=publish, args=('b', b'second payload'), name='publisher-b')
    first.start()
    second.start()
    first.join(15)
    second.join(15)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert results == {'a': True, 'b': False}
    assert destination.read_bytes() == b'first payload'
    assert list(tmp_path.iterdir()) == [destination]
