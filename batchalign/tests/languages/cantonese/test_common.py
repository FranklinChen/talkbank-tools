"""Unit tests for _common.py: Cantonese normalization, config, timestamps."""

from __future__ import annotations

import configparser

import pytest

from batchalign.errors import ConfigError
from batchalign.inference.languages.cantonese._common import (
    parse_timestamp_pair,
    read_asr_config,
)

# Cantonese normalization is not tested here, because Python no longer performs
# any. Its one owner is `AlignedNormalization` in `batchalign-transform`, tested
# in `crates/batchalign-transform/src/asr_postprocess/cantonese.rs`.


# ---------------------------------------------------------------------------
# parse_timestamp_pair
# ---------------------------------------------------------------------------


class TestParseTimestampPair:
    def test_normal_ints(self) -> None:
        assert parse_timestamp_pair([100, 200]) == (100, 200)

    def test_float_rounding(self) -> None:
        assert parse_timestamp_pair([100.4, 210.6]) == (100, 211)

    def test_none_input(self) -> None:
        assert parse_timestamp_pair(None) == (None, None)

    def test_non_numeric(self) -> None:
        assert parse_timestamp_pair(["x", "y"]) == (None, None)

    def test_short_list(self) -> None:
        assert parse_timestamp_pair([100]) == (None, None)

    def test_empty_list(self) -> None:
        assert parse_timestamp_pair([]) == (None, None)

    def test_string_input(self) -> None:
        assert parse_timestamp_pair("ab") == (None, None)

    def test_tuple_input(self) -> None:
        assert parse_timestamp_pair((50, 100)) == (50, 100)

    def test_zero_values(self) -> None:
        assert parse_timestamp_pair([0, 0]) == (0, 0)

    def test_large_values(self) -> None:
        assert parse_timestamp_pair([3600000, 3600500]) == (3600000, 3600500)


# ---------------------------------------------------------------------------
# read_asr_config
# ---------------------------------------------------------------------------


def config_with_asr(**entries: str) -> configparser.ConfigParser:
    """Build a minimal `[asr]` config fixture for HK config tests."""
    cfg = configparser.ConfigParser()
    cfg.add_section("asr")
    for key, value in entries.items():
        cfg.set("asr", key, value)
    return cfg


class TestReadAsrConfig:
    def test_missing_asr_section(self) -> None:
        with pytest.raises(ConfigError):
            read_asr_config(
                ("engine.tencent.id",),
                engine="Tencent",
                config=configparser.ConfigParser(),
            )

    def test_missing_keys(self) -> None:
        with pytest.raises(ConfigError):
            read_asr_config(
                ("engine.tencent.id", "engine.tencent.key"),
                engine="Tencent",
                config=config_with_asr(),
            )

    def test_empty_value(self) -> None:
        with pytest.raises(ConfigError):
            read_asr_config(
                ("engine.tencent.id",),
                engine="Tencent",
                config=config_with_asr(**{"engine.tencent.id": "   "}),
            )

    def test_valid_config(self) -> None:
        values = read_asr_config(
            ("engine.tencent.id", "engine.tencent.key"),
            engine="Tencent",
            config=config_with_asr(
                **{
                    "engine.tencent.id": " my_id ",
                    "engine.tencent.key": " my_key ",
                }
            ),
        )
        assert values["engine.tencent.id"] == "my_id"
        assert values["engine.tencent.key"] == "my_key"

    def test_injected_env_overrides_config_file_reads(self) -> None:
        values = read_asr_config(
            ("engine.tencent.id", "engine.tencent.key"),
            engine="Tencent",
            config=configparser.ConfigParser(),
            environ={
                "BATCHALIGN_TENCENT_ID": " env-id ",
                "BATCHALIGN_TENCENT_KEY": " env-key ",
            },
        )
        assert values["engine.tencent.id"] == "env-id"
        assert values["engine.tencent.key"] == "env-key"

    def test_aliyun_keys(self) -> None:
        values = read_asr_config(
            (
                "engine.aliyun.ak_id",
                "engine.aliyun.ak_secret",
                "engine.aliyun.ak_appkey",
            ),
            engine="Aliyun",
            config=config_with_asr(
                **{
                    "engine.aliyun.ak_id": "ak_id_val",
                    "engine.aliyun.ak_secret": "ak_secret_val",
                    "engine.aliyun.ak_appkey": "ak_appkey_val",
                }
            ),
        )
        assert len(values) == 3
        assert values["engine.aliyun.ak_id"] == "ak_id_val"
