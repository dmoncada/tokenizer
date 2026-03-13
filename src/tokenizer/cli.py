"""Simple CLI tool for OpenAI's `tiktoken`.

This tool tokenizes text from a file or standard input and can output:
- The total token count
- CSV of token IDs and decoded tokens
- JSONL of token IDs and decoded tokens
"""

import csv
import json
import signal
import sys
from enum import StrEnum
from typing import Annotated, Generator, TextIO, Tuple

import tiktoken
import typer

app = typer.Typer()


class OutputFormat(StrEnum):
    """Supported output formats for the CLI.

    Attributes:
        token_count: Output only the total number of tokens.
        csv: Output a CSV stream of token IDs and decoded tokens.
        jsonl: Output a JSON Lines stream of token IDs and decoded tokens.
    """

    token_count = "count"
    csv = "csv"
    jsonl = "jsonl"


def get_encoding(name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for a model or encoding name.

    Tries to resolve the encoding via `encoding_for_model` first. If the
    provided name is not a recognized model, falls back to `get_encoding`.

    Args:
        name: The model name or encoding name.

    Returns:
        A tiktoken.Encoding instance.
    """
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        return tiktoken.get_encoding(name)


def tokenize_stream(
    lines: TextIO, model: str, decode: bool
) -> Generator[Tuple[int, str], None, None]:
    """Tokenize input lines as a stream.

    Yields token IDs and optionally the decoded token strings.

    Args:
        lines: A file-like object to read lines from.
        model: Model or encoding name for tokenization.
        decode: If True, also yield the decoded token strings.

    Yields:
        Tuples of (token_id, token_str). If decode=False, token_str is empty.
    """
    encoding = get_encoding(model)
    for line in lines:
        ids = encoding.encode(line)
        tokens = [encoding.decode([tid]) for tid in ids] if decode else []
        for i, tid in enumerate(ids):
            tok = tokens[i] if decode else ""
            yield tid, tok


def sanitize(token: str) -> str:
    r"""Escape newline and carriage return characters in a token string.

    Args:
        token: The token string to sanitize.

    Returns:
        A string with newlines and carriage returns replaced by \n and \r.
    """
    return token.replace("\n", "\\n").replace("\r", "\\r")


def output_csv_stream(items: Generator[Tuple[int, str], None, None]):
    """Write a stream of tokens to stdout in CSV format.

    The CSV includes a header row ('id', 'token'). Tokens are sanitized.

    Args:
        items: A generator of (token_id, token_str) tuples.
    """
    writer = csv.writer(typer.get_text_stream("stdout"))
    writer.writerow(["id", "token"])
    for tid, tok in items:
        writer.writerow([tid, sanitize(tok)])


def output_jsonl_stream(items: Generator[Tuple[int, str], None, None]):
    """Write a stream of tokens to stdout in JSON Lines format.

    Each line contains an object with 'id' and 'token' fields. Tokens are sanitized.

    Args:
        items: A generator of (token_id, token_str) tuples.
    """
    for tid, tok in items:
        typer.echo(json.dumps({"id": tid, "token": sanitize(tok)}))


@app.command()
def main(
    file: Annotated[
        typer.FileText,
        typer.Argument(default_factory=lambda: sys.stdin, show_default="sys.stdin"),
    ],
    model: str = typer.Option(
        "gpt-5", "--model", "-m", help="Model/encoding for tokenization."
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.token_count,
        "--format",
        "-f",
        help="Output format: count, csv, or jsonl.",
    ),
):
    """Tokenize text from a file or standard input.

    Depending on the selected output format, prints either:
    - The total token count
    - A CSV stream of token IDs and decoded tokens
    - A JSONL stream of token IDs and decoded tokens

    Args:
        file: File-like object to read text from, or stdin if None.
        model: Model/encoding name to use for tokenization.
        format: Output format (token count, CSV, or JSONL).
    """
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    decode = format != OutputFormat.token_count
    stream = tokenize_stream(file, model, decode)

    if format == OutputFormat.token_count:
        typer.echo(sum(1 for _ in stream))
    elif format == OutputFormat.csv:
        output_csv_stream(stream)
    elif format == OutputFormat.jsonl:
        output_jsonl_stream(stream)


if __name__ == "__main__":
    app()
