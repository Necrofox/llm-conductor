"""Platform-specific directory synchronization without filesystem side effects."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_conductor.core import io as io_module
from llm_conductor.core.io import FileOperations


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_directory_sync_is_linux_only(monkeypatch, platform):
    monkeypatch.setattr(io_module, "sys", SimpleNamespace(platform=platform))
    open_directory = Mock(side_effect=AssertionError("Directory handles are not portable"))
    monkeypatch.setattr(io_module, "os", SimpleNamespace(open=open_directory))
    FileOperations._sync_parent_directory("parent")
    open_directory.assert_not_called()


@pytest.mark.parametrize("failure", [None, "open", "fsync"])
def test_linux_directory_handle_is_synchronized_and_closed(monkeypatch, failure):
    events = []

    def open_directory(path, flags):
        events.append(("open", path, flags))
        if failure == "open":
            raise OSError("open failed")
        return 42

    def sync_directory(fd):
        events.append(("fsync", fd))
        if failure == "fsync":
            raise OSError("fsync failed")

    monkeypatch.setattr(io_module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(io_module, "os", SimpleNamespace(
        O_RDONLY=0, O_DIRECTORY=0x10000, open=open_directory,
        fsync=sync_directory, close=lambda fd: events.append(("close", fd)),
    ))
    if failure:
        with pytest.raises(OSError, match=f"{failure} failed"):
            FileOperations._sync_parent_directory("parent")
    else:
        FileOperations._sync_parent_directory("parent")
    assert events[0] == ("open", "parent", 0x10000)
    assert events[1:] == ([] if failure == "open" else [("fsync", 42), ("close", 42)])
