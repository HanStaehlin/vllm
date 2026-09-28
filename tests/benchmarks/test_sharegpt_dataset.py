# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ShareGPT benchmark dataset.

Uses a whitespace tokenizer so the token counts are exact and no model
download is needed.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from vllm.benchmarks.datasets import ShareGPTDataset

pytestmark = pytest.mark.skip_global_cleanup


class _TokenizedPrompt:
    def __init__(self, prompt: str) -> None:
        self.input_ids = prompt.split()


class _Tokenizer:
    def __call__(self, prompt: str) -> _TokenizedPrompt:
        return _TokenizedPrompt(prompt)


def _words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


def _write(path: Path, entries: list[dict[str, Any]]) -> str:
    path.write_text(json.dumps(entries))
    return str(path)


def _conv(conv_id: str, *turns: tuple[str, str]) -> dict[str, Any]:
    return {
        "id": conv_id,
        "conversations": [{"from": role, "value": value} for role, value in turns],
    }


def _sample(path: str, num_requests: int, **kwargs: Any) -> ShareGPTDataset:
    dataset = ShareGPTDataset(dataset_path=path, random_seed=0, disable_shuffle=True)
    dataset.samples = dataset.sample(
        tokenizer=_Tokenizer(),
        num_requests=num_requests,
        no_oversample=True,
        **kwargs,
    )
    return dataset


def test_role_pairing_skips_leading_assistant_turn(tmp_path: Path) -> None:
    """A conversation part that opens with 'gpt' must not use it as prompt."""
    path = _write(
        tmp_path / "sg.json",
        [
            _conv(
                "a_1",
                ("gpt", _words(8, "reply")),
                ("human", _words(6, "question")),
                ("gpt", _words(5, "answer")),
            )
        ],
    )
    dataset = _sample(path, num_requests=1)
    [req] = dataset.samples
    assert req.prompt == _words(6, "question")
    assert req.prompt_len == 6
    assert req.expected_output_len == 5
    assert dataset.num_skipped_no_pair == 0


def test_role_pairing_skips_leading_system_turn(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "sg.json",
        [
            _conv(
                "s_0",
                ("system", _words(10, "sys")),
                ("user", _words(6, "question")),
                ("assistant", _words(5, "answer")),
            )
        ],
    )
    [req] = _sample(path, num_requests=1).samples
    assert req.prompt == _words(6, "question")
    assert req.expected_output_len == 5


@pytest.mark.parametrize("assistant_role", ["gpt", "chatgpt", "bing", "bard"])
def test_role_pairing_accepts_sharegpt_assistant_roles(
    tmp_path: Path, assistant_role: str
) -> None:
    path = _write(
        tmp_path / "sg.json",
        [_conv("r_0", ("human", _words(6)), (assistant_role, _words(5)))],
    )
    assert len(_sample(path, num_requests=1).samples) == 1


def test_role_pairing_positional_case_unchanged(tmp_path: Path) -> None:
    """human-first conversations still yield conversations[0]/[1]."""
    path = _write(
        tmp_path / "sg.json",
        [
            _conv(
                "p_0",
                ("human", _words(6, "q1")),
                ("gpt", _words(5, "a1")),
                ("human", _words(7, "q2")),
                ("gpt", _words(9, "a2")),
            )
        ],
    )
    [req] = _sample(path, num_requests=1).samples
    assert req.prompt == _words(6, "q1")
    assert req.expected_output_len == 5


def test_role_pairing_skips_and_counts_conversations_without_pair(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "sg.json",
        [
            # assistant then user: no user turn followed by an assistant turn
            _conv("n_0", ("gpt", _words(5)), ("human", _words(6))),
            # two user turns, no assistant
            _conv("n_1", ("human", _words(5)), ("human", _words(6))),
            # user then system
            _conv("n_2", ("human", _words(5)), ("system", _words(6))),
            _conv("ok_0", ("human", _words(6, "q")), ("gpt", _words(5, "a"))),
        ],
    )
    dataset = _sample(path, num_requests=10)
    assert [r.prompt for r in dataset.samples] == [_words(6, "q")]
    assert dataset.num_skipped_no_pair == 3


# --- --sharegpt-max-prompt-len / --sharegpt-max-total-len --------------------


def _length_fixture(tmp_path: Path) -> str:
    return _write(
        tmp_path / "sg.json",
        [
            # prompt 100, output 50 -> fits the defaults (1024 / 2048)
            _conv("small_0", ("human", _words(100)), ("gpt", _words(50))),
            # prompt 1500, output 100 -> over the default prompt cap only
            _conv("long_prompt_0", ("human", _words(1500)), ("gpt", _words(100))),
            # prompt 1500, output 1000 -> over both default caps
            _conv("long_total_0", ("human", _words(1500)), ("gpt", _words(1000))),
        ],
    )


def _prompt_lens(path: str, **kwargs: Any) -> list[int]:
    return [r.prompt_len for r in _sample(path, num_requests=10, **kwargs).samples]


def test_caps_defaults_unchanged(tmp_path: Path) -> None:
    assert _prompt_lens(_length_fixture(tmp_path)) == [100]


def test_caps_prompt_only_trap(tmp_path: Path) -> None:
    """Raising only max_prompt_len still enforces the default max_total_len."""
    path = _length_fixture(tmp_path)
    # 1500 + 100 = 1600 <= 2048 is kept; 1500 + 1000 = 2500 > 2048 is not.
    assert _prompt_lens(path, max_prompt_len=4096) == [100, 1500]


def test_caps_both_limits(tmp_path: Path) -> None:
    path = _length_fixture(tmp_path)
    assert _prompt_lens(path, max_prompt_len=4096, max_total_len=4096) == [
        100,
        1500,
        1500,
    ]
    # A tight total cap drops requests the prompt cap alone would keep.
    assert _prompt_lens(path, max_prompt_len=4096, max_total_len=120) == []


def _serve_parser():
    from vllm.benchmarks.datasets import add_dataset_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    return parser


def test_caps_cli_defaults_and_help() -> None:
    parser = _serve_parser()
    args = parser.parse_args([])
    assert args.sharegpt_max_prompt_len == 1024
    assert args.sharegpt_max_total_len == 2048

    helps = {
        opt: action.help for action in parser._actions for opt in action.option_strings
    }
    # Each help text must say that both limits apply.
    assert "--sharegpt-max-total-len" in helps["--sharegpt-max-prompt-len"]
    assert "--sharegpt-max-prompt-len" in helps["--sharegpt-max-total-len"]


def test_caps_wired_through_get_samples(tmp_path: Path) -> None:
    from vllm.benchmarks.datasets import get_samples

    path = _length_fixture(tmp_path)
    args = _serve_parser().parse_args(
        [
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            path,
            "--num-prompts",
            "10",
            "--no-oversample",
            "--disable-shuffle",
            "--sharegpt-max-prompt-len",
            "4096",
            "--sharegpt-max-total-len",
            "4096",
        ]
    )
    args.backend = "openai"
    samples = get_samples(args, _Tokenizer())
    assert [s.prompt_len for s in samples] == [100, 1500, 1500]


def test_throughput_namespace_without_cap_attrs(tmp_path: Path) -> None:
    """`vllm bench throughput` builds its own namespace for get_samples.

    It does not define the ShareGPT cap attributes, so get_samples must fall
    back to the defaults instead of raising AttributeError.
    """
    from vllm.benchmarks.throughput import add_cli_args, get_requests
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            _length_fixture(tmp_path),
            "--num-prompts",
            "1",
            "--backend",
            "vllm",
        ]
    )
    assert not hasattr(args, "sharegpt_max_prompt_len")
    assert not hasattr(args, "sharegpt_max_total_len")

    requests = get_requests(args, _Tokenizer())
    assert [r.prompt_len for r in requests] == [100]
