from pathlib import Path

import pytest

from src.exceptions import (
    DirectoryCreationError,
    FileCopyError,
    FileManagerError,
    FileMoveError,
    FileReadError,
    FileWriteError,
)
from src.utils.file_manager import FileManager


@pytest.fixture
def manager() -> FileManager:
    return FileManager()


def test_create_directory(manager, tmp_path):
    directory = tmp_path / "data"

    result = manager.create_directory(directory)

    assert result == directory
    assert directory.exists()
    assert directory.is_dir()


def test_create_directory_existing_is_idempotent(manager, tmp_path):
    directory = tmp_path / "data"

    manager.create_directory(directory)
    manager.create_directory(directory)

    assert directory.exists()


def test_file_exists(manager, tmp_path):
    file = tmp_path / "sample.txt"

    file.write_text("hello")

    assert manager.file_exists(file)
    assert not manager.file_exists(tmp_path / "missing.txt")


def test_directory_exists(manager, tmp_path):
    directory = tmp_path / "logs"

    manager.create_directory(directory)

    assert manager.directory_exists(directory)
    assert not manager.directory_exists(tmp_path / "missing")


def test_copy_file(manager, tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "copy" / "a.txt"

    source.write_text("hello")

    result = manager.copy_file(source, destination)

    assert result == destination
    assert destination.exists()
    assert destination.read_text() == "hello"


def test_copy_missing_source_raises(manager, tmp_path):
    with pytest.raises(FileReadError):
        manager.copy_file(tmp_path / "missing.txt", tmp_path / "out.txt")


def test_move_file(manager, tmp_path):
    source = tmp_path / "move.txt"
    destination = tmp_path / "new" / "move.txt"

    source.write_text("hello")

    result = manager.move_file(source, destination)

    assert result == destination
    assert destination.exists()
    assert not source.exists()


def test_move_missing_source_raises(manager, tmp_path):
    with pytest.raises(FileReadError):
        manager.move_file(tmp_path / "missing.txt", tmp_path / "out.txt")


def test_delete_file(manager, tmp_path):
    file = tmp_path / "delete.txt"

    file.write_text("hello")

    manager.delete_file(file)

    assert not file.exists()


def test_delete_missing_file_is_noop(manager, tmp_path):
    manager.delete_file(tmp_path / "missing.txt")


def test_delete_directory(manager, tmp_path):
    directory = tmp_path / "tree" / "nested"

    manager.create_directory(directory)

    manager.delete_directory(tmp_path / "tree")

    assert not (tmp_path / "tree").exists()


def test_list_files(manager, tmp_path):
    (tmp_path / "a.csv").write_text("")
    (tmp_path / "b.csv").write_text("")
    (tmp_path / "c.txt").write_text("")

    files = manager.list_files(tmp_path, "*.csv")

    assert files == [tmp_path / "a.csv", tmp_path / "b.csv"]


def test_get_file_size(manager, tmp_path):
    file = tmp_path / "size.txt"

    file.write_text("abcdef")

    assert manager.get_file_size(file) == 6


def test_get_file_size_missing_raises(manager, tmp_path):
    with pytest.raises(FileReadError):
        manager.get_file_size(tmp_path / "missing.txt")


def test_read_write_yaml_round_trip(manager, tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    data = {"name": "test", "values": [1, 2, 3]}

    manager.write_yaml(path, data)

    assert manager.read_yaml(path) == data


def test_read_yaml_missing_raises(manager, tmp_path):
    with pytest.raises(FileReadError):
        manager.read_yaml(tmp_path / "missing.yaml")


def test_read_write_csv_round_trip(manager, tmp_path):
    path = tmp_path / "data.csv"
    rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    manager.write_csv(path, rows, fieldnames=["a", "b"])

    assert manager.read_csv(path) == rows


def test_read_csv_missing_raises(manager, tmp_path):
    with pytest.raises(FileReadError):
        manager.read_csv(tmp_path / "missing.csv")


def test_exception_types_are_project_errors():
    assert issubclass(DirectoryCreationError, FileManagerError)
    assert issubclass(FileCopyError, FileManagerError)
    assert issubclass(FileMoveError, FileManagerError)
    assert issubclass(FileReadError, FileManagerError)
    assert issubclass(FileWriteError, FileManagerError)
    assert issubclass(FileManagerError, Exception)


def test_directory_creation_error_message(manager, tmp_path):
    blocker = tmp_path / "blocker"

    blocker.write_text("")

    with pytest.raises(DirectoryCreationError, match="directory"):
        manager.create_directory(blocker / "nested")


def test_path_helpers_accept_strings(manager, tmp_path):
    file = tmp_path / "str.txt"

    file.write_text("x")

    assert manager.file_exists(str(file))
    assert manager.get_file_size(str(file)) == 1

    _ = Path(tmp_path)
