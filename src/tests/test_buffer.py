"""This is part of the MSS Python's module.
Source: https://github.com/BoboTiG/python-mss.
"""

import gc

import numpy as np
import pytest

from mss.buffer import FAST_PATH_AVAILABLE, _FinalizingBufferIntermediate, finalizing_buffer


def test_finalizing_buffer_preserves_readonly_and_returns_memoryview() -> None:
    writable = bytearray(b"abcd")
    writable_called = 0

    def writable_finalizer() -> None:
        nonlocal writable_called
        writable_called += 1

    writable_view = finalizing_buffer(writable, writable_finalizer)
    assert isinstance(writable_view, memoryview)
    assert not writable_view.readonly

    readonly = b"abcd"
    readonly_called = 0

    def readonly_finalizer() -> None:
        nonlocal readonly_called
        readonly_called += 1

    readonly_view = finalizing_buffer(readonly, readonly_finalizer)
    assert isinstance(readonly_view, memoryview)
    assert readonly_view.readonly

    writable_view.release()
    readonly_view.release()
    gc.collect()

    assert writable_called == 1
    assert readonly_called == 1


@pytest.mark.skipif(FAST_PATH_AVAILABLE, reason="Covers behavior only present before Python 3.12")
def test_finalizing_buffer_slow_path_copies_and_finalizes_immediately() -> None:
    data = bytearray(b"abcd")
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    wrapped = finalizing_buffer(data, finalizer)
    assert finalizer_calls == 1

    data[0] = ord("Z")
    assert wrapped.tobytes() == b"abcd"

    wrapped[1] = ord("Y")
    assert data == bytearray(b"Zbcd")

    wrapped.release()
    gc.collect()
    assert finalizer_calls == 1


@pytest.mark.skipif(not FAST_PATH_AVAILABLE, reason="Covers behavior only present in Python 3.12+")
def test_finalizing_buffer_fast_path_is_zero_copy() -> None:
    data = bytearray(b"abcd")
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    wrapped = finalizing_buffer(data, finalizer)
    assert finalizer_calls == 0

    data[0] = ord("Z")
    assert wrapped[0] == ord("Z")

    wrapped[1] = ord("Y")
    assert data[1] == ord("Y")

    wrapped.release()
    gc.collect()
    assert finalizer_calls == 1


@pytest.mark.skipif(not FAST_PATH_AVAILABLE, reason="Covers behavior only present in Python 3.12+")
def test_child_memoryview_defers_finalizer_until_child_release() -> None:
    data = bytearray(b"abcdefgh")
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    parent = finalizing_buffer(data, finalizer)
    child = memoryview(parent)

    del parent
    gc.collect()
    assert finalizer_calls == 0

    child.release()
    del child
    gc.collect()
    assert finalizer_calls == 1


@pytest.mark.skipif(not FAST_PATH_AVAILABLE, reason="Covers behavior only present in Python 3.12+")
def test_numpy_frombuffer_child_defers_finalizer_until_array_deleted() -> None:
    data = bytearray(b"abcdefgh")
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    parent = finalizing_buffer(data, finalizer)
    array = np.frombuffer(parent, dtype=np.uint8)

    del parent
    gc.collect()
    assert finalizer_calls == 0

    del array
    gc.collect()
    assert finalizer_calls == 1


def test_finalizer_runs_once() -> None:
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    wrapped = finalizing_buffer(bytearray(b"abcd"), finalizer)
    assert finalizer_calls == 0

    del wrapped
    gc.collect()
    assert finalizer_calls == 1


@pytest.mark.skipif(not FAST_PATH_AVAILABLE, reason="Covers behavior only present in Python 3.12+")
def test_intermediate_allows_single_buffer_request_and_release() -> None:
    finalizer_calls = 0

    def finalizer() -> None:
        nonlocal finalizer_calls
        finalizer_calls += 1

    intermediate = _FinalizingBufferIntermediate(bytearray(b"abcd"), finalizer)

    view = intermediate.__buffer__(0)
    assert view.tobytes() == b"abcd"

    with pytest.raises(AssertionError, match="Buffer can only be requested once"):
        intermediate.__buffer__(0)

    intermediate.__release_buffer__(view)
    assert finalizer_calls == 1

    with pytest.raises(AssertionError, match="Buffer can only be released once"):
        intermediate.__release_buffer__(view)
