"""The Stage B spill's device input digest (PQ #1030).

The spill compares each later probe's input with probe 0's. It used to hash
every input with SHA-256 on the writer thread, after copying it to the host;
it now digests the operand on its own device when the hook fires. These tests
pin the digest's arithmetic against a pure-Python reference and show that
one flipped bit anywhere changes it.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant import joint_replay_spill as spill_mod

P = spill_mod._DIGEST_PRIME
SEGMENT = spill_mod._DIGEST_SEGMENT_WORDS
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def digests():
    return {device: spill_mod._InputDigest(device) for device in DEVICES}


def _reference(digest, tensor):
    """The digest in Python integers, from its definition."""
    words = spill_mod._storage_order(tensor.cpu()).view(torch.int16).tolist()
    word_keys = digest.word_keys.cpu().tolist()
    segment_keys = digest.segment_keys.cpu().tolist()
    result = []
    for k, r in zip(word_keys, segment_keys):
        total = 0
        for start in range(0, len(words), SEGMENT):
            segment = words[start:start + SEGMENT]
            inner = sum((w & 0xFFFF) * k[i] for i, w in enumerate(segment)) % P
            total += inner * r[start // SEGMENT]
        result.append(total % P)
    return result


def _tensor(numel, *, seed=0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(numel, generator=generator) * 3).to(dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("numel", [1, 7, SEGMENT - 1, SEGMENT, SEGMENT + 3,
                                   spill_mod._DIGEST_BLOCK_SEGMENTS * SEGMENT + 5])
def test_the_digest_is_its_definition(digests, device, numel):
    tensor = _tensor(numel, seed=numel).to(device)
    assert digests[device](tensor).cpu().tolist() == _reference(digests[device], tensor)


def test_the_digest_does_not_depend_on_the_device(digests):
    if "cuda" not in digests:
        pytest.skip("no CUDA device on this worker")
    tensor = _tensor(3 * SEGMENT + 11, seed=5)
    assert torch.equal(digests["cpu"](tensor), digests["cuda"](tensor.cuda()).cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_one_flipped_bit_anywhere_changes_both_digests(digests, device):
    digest = digests[device]
    tensor = _tensor(2 * SEGMENT + 9, seed=11).to(device)
    base = digest(tensor)
    words = tensor.numel()
    for position in (0, 1, SEGMENT - 1, SEGMENT, 2 * SEGMENT, words - 1):
        for bit in (0, 7, 15):
            flipped = tensor.clone()
            flipped.view(torch.int16)[position] ^= (1 << bit) if bit < 15 else -(1 << 15)
            changed = digest(flipped)
            assert (changed != base).all(), (position, bit)


@pytest.mark.parametrize("device", DEVICES)
def test_moved_words_change_the_digest(digests, device):
    """Unlike a plain sum, the digest sees where each word is."""
    digest = digests[device]
    tensor = _tensor(SEGMENT + 17, seed=3).to(device)
    swapped = tensor.clone()
    swapped[[0, SEGMENT]] = tensor[[SEGMENT, 0]]
    assert not torch.equal(tensor[0], tensor[SEGMENT])
    assert (digest(swapped) != digest(tensor)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_the_digest_reads_storage_order_and_every_byte(digests, device):
    digest = digests[device]
    matrix = _tensor(64 * 48, seed=9).view(64, 48).to(device)
    # A transposed view is the same storage in the same order.
    assert torch.equal(digest(matrix.t()), digest(matrix))
    # Four-byte elements are two words each.
    wide = _tensor(1000, seed=4, dtype=torch.float32).to(device)
    assert digest(wide).cpu().tolist() == _reference(digest, wide)
    flipped = wide.clone()
    flipped.view(torch.int16)[1] ^= 1
    assert (digest(flipped) != digest(wide)).all()


def test_the_keys_are_fixed():
    first, second = spill_mod._InputDigest("cpu"), spill_mod._InputDigest("cpu")
    assert torch.equal(first.word_keys, second.word_keys)
    assert torch.equal(first.segment_keys, second.segment_keys)
    assert not torch.equal(first.word_keys[0], first.word_keys[1])
    assert int(first.word_keys.min()) >= 1 and int(first.word_keys.max()) < P
