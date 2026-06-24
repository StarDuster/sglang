"""Streaming detokenizer regression: sent_offset rewound on a mid-multibyte flush,
duplicating a CJK character right before a following emoji ("快🔥🔥" -> "快快🔥🔥").

Model-free: a byte-per-token tokenizer plus a step=2 flush makes a flush land
inside a multi-byte character (the trigger). The test drives the real
DetokenizerManager._decode_batch_token_id_output; RED pre-fix, GREEN after.
"""

import unittest
from types import MethodType, SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# CJK+emoji strings that duplicated the CJK part on the pre-fix code, e.g.
# "太强了🎉🎉" was emitted as "太强了了🎉🎉".
_REPROS = ["太强了🎉🎉", "快🔥🔥", "强🚀🚀", "牛🎂🎂"]


class ByteTokenizer:
    # token id == one UTF-8 byte; decode = byte concat + utf-8 "replace".
    all_special_ids_set = set()

    def decode(self, ids, **kw):
        return bytes(ids).decode("utf-8", "replace")


def _decode_stream(text, step=2):
    """Stream `text` through the real detok, flushing `step` bytes per step."""
    s = SimpleNamespace(
        disable_tokenizer_batch_decode=False,
        is_tool_call_parser_gpt_oss=False,
        tokenizer=ByteTokenizer(),
        decode_status={},
    )
    s.trim_matched_stop = MethodType(DetokenizerManager.trim_matched_stop, s)
    s._grouped_batch_decode = MethodType(DetokenizerManager._grouped_batch_decode, s)

    ids = list(text.encode())
    out = []
    prev = 0
    bounds = list(range(step, len(ids), step)) + [len(ids)]
    for i, sp in enumerate(bounds):
        recv = SimpleNamespace(
            rids=["r"],
            decoded_texts=[""],
            decode_ids=[ids[prev:sp]],
            read_offsets=[0],
            finished_reasons=[{} if i == len(bounds) - 1 else None],
            no_stop_trim=[False],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[True],
        )
        out.append(DetokenizerManager._decode_batch_token_id_output(s, recv)[0])
        prev = sp
    return "".join(out)


class TestSentOffsetRewind(unittest.TestCase):
    def test_no_duplicate_cjk_before_emoji(self):
        for text in _REPROS:
            self.assertEqual(_decode_stream(text), text, f"duplicated: {text!r}")


if __name__ == "__main__":
    unittest.main()
