# Tokenizer

Simple CLI tool for OpenAI's [`tiktoken`][1].

## Installation

```bash
uv tool install git+https://github.com/dmoncada/tokenizer
```

## Usage

```bash
# Print help:
tokenizer --help

# Print count:
tokenizer README.md

# Output:
# 337

# Input from file; pipe, for ex. to `csvlook`:
tokenizer README.md -f csv | csvlook | head

# Output:
# |      id | token         |
# | ------- | ------------- |
# |       2 | #             |
# |  17,951 |  Token        |
# |   4,492 | izer          |
# |     198 | \n            |
# |     198 | \n            |
# |  17,958 | Simple        |
# |  83,887 |  CLI          |
# |   4,584 |  tool         |

# Input from stdin; pipe, for ex. to `jq`:
cat README.md | tokenizer -f jsonl | jq | head -8

# Output:
# {
#   "id": 2,
#   "token": "#"
# }
# {
#   "id": 17951,
#   "token": " Token"
# }
```

## See also

- [OpenAI Platform Tokenizer][2]

[1]: https://github.com/openai/tiktoken
[2]: https://platform.openai.com/tokenizer
