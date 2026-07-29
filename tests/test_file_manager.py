from pathlib import Path

from src.utils.file_manager import FileManager


def test_create_directory(tmp_path):
    directory = tmp_path / "data"

    FileManager.create_directory(directory)

    assert directory.exists()
    assert directory.is_dir()


def test_touch_file(tmp_path):
    file = tmp_path / "sample.txt"

    FileManager.touch(file)

    assert file.exists()


def test_file_exists(tmp_path):
    file = tmp_path / "sample.txt"

    FileManager.touch(file)

    assert FileManager.file_exists(file)


def test_directory_exists(tmp_path):
    directory = tmp_path / "logs"

    FileManager.create_directory(directory)

    assert FileManager.directory_exists(directory)


def test_copy_file(tmp_path):
    source = tmp_path / "a.txt"
    destination = tmp_path / "copy" / "a.txt"

    source.write_text("hello")

    FileManager.copy_file(source, destination)

    assert destination.exists()
    assert destination.read_text() == "hello"


def test_move_file(tmp_path):
    source = tmp_path / "move.txt"
    destination = tmp_path / "new" / "move.txt"

    source.write_text("hello")

    FileManager.move_file(source, destination)

    assert destination.exists()
    assert not source.exists()


def test_delete_file(tmp_path):
    file = tmp_path / "delete.txt"

    file.write_text("hello")

    FileManager.delete_file(file)

    assert not file.exists()


def test_list_files(tmp_path):
    (tmp_path / "a.csv").write_text("")
    (tmp_path / "b.csv").write_text("")
    (tmp_path / "c.txt").write_text("")

    files = FileManager.list_files(tmp_path, "*.csv")

    assert len(files) == 2


def test_file_size(tmp_path):
    file = tmp_path / "size.txt"

    file.write_text("abcdef")

    assert FileManager.file_size(file) == 6
