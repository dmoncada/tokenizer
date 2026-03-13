"""End-to-end tests for the CLI tool using a deterministic continuous-token mock."""

from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tokenizer.cli import app

INPUT_TEXT = "Hello, world!"

runner = CliRunner()


class MockEncoding:
    """Mock tiktoken encoding with continuous token IDs across lines."""

    counter: int = 0

    def encode(self, text: str) -> List[int]:
        """Return a continuous sequence of token IDs for all words."""
        ids: list[int] = []
        for _ in text.split():
            self.counter += 1
            ids.append(self.counter)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Return a deterministic decoded token for each ID."""
        return f"token{ids[0]}"


@pytest.mark.parametrize("format_option", ["count", "csv", "jsonl"])
def test_cli_with_file_mocked(tmp_path: Path, format_option: str) -> None:
    """Test CLI with file input using continuous token mock.

    Args:
        tmp_path: Temporary directory fixture from pytest.
        format_option: Output format option ("count", "csv", "jsonl").
    """
    file_path = tmp_path / "input.txt"
    file_path.write_text(INPUT_TEXT, encoding="utf-8")

    with patch("tokenizer.cli.get_encoding", return_value=MockEncoding()):
        result = runner.invoke(app, [str(file_path), "-f", format_option])
        assert result.exit_code == 0

        token_count = sum(len(line.split()) for line in INPUT_TEXT.splitlines())
        if format_option == "count":
            assert str(token_count) in result.output
        if format_option == "csv":
            assert "id,token" in result.output
            for i in range(1, token_count + 1):
                assert f"token{i}" in result.output
        if format_option == "jsonl":
            for i in range(1, token_count + 1):
                assert f'"token": "token{i}"' in result.output


@pytest.mark.parametrize("format_option", ["count", "csv", "jsonl"])
def test_cli_with_stdin_mocked(format_option: str) -> None:
    """Test CLI with stdin input using continuous token mock.

    Args:
        format_option: Output format option ("count", "csv", "jsonl").
    """
    with patch("tokenizer.cli.get_encoding", return_value=MockEncoding()):
        result = runner.invoke(app, ["-f", format_option], input=INPUT_TEXT)
        assert result.exit_code == 0

        token_count = sum(len(line.split()) for line in INPUT_TEXT.splitlines())
        if format_option == "count":
            assert str(token_count) in result.output
        if format_option == "csv":
            assert "id,token" in result.output
            for i in range(1, token_count + 1):
                assert f"token{i}" in result.output
        if format_option == "jsonl":
            for i in range(1, token_count + 1):
                assert f'"token": "token{i}"' in result.output


def test_cli_nonexistent_file() -> None:
    """Test that the CLI exits with an error when the input file does not exist."""
    nonexistent_path = "nonexistent_file.txt"
    result = runner.invoke(app, [nonexistent_path])
    assert result.exit_code != 0
