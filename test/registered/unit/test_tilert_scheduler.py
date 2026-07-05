from sglang.srt.managers.tilert_utils import cached_len_after_external_prefill_tail
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_external_prefill_tail_disabled_keeps_cached_len():
    assert (
        cached_len_after_external_prefill_tail(
            64960,
            64980,
            0,
            with_mtp=True,
            mtp_seq_len=4,
        )
        == 64960
    )


def test_external_prefill_tail_aligns_mtp_cached_len():
    assert (
        cached_len_after_external_prefill_tail(
            64960,
            64980,
            1023,
            with_mtp=True,
            mtp_seq_len=4,
        )
        == 63936
    )


def test_external_prefill_tail_keeps_at_least_one_cached_token():
    assert (
        cached_len_after_external_prefill_tail(
            3,
            10,
            4096,
            with_mtp=True,
            mtp_seq_len=4,
        )
        == 1
    )
