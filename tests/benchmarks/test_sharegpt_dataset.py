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
