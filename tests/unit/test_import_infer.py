import sys, os, json, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from zoneinfo import ZoneInfo

import pytest

from backfill.import_infer import (
    DeviceMap,
    merge_windows,
    parse_ts,
    read_csv_rows,
    read_infer_json,
)
from modules.inference.model import BehaviorLabel

SH = ZoneInfo("Asia/Shanghai")
SLEEP = int(BehaviorLabel.SLEEP)
MOVE = int(BehaviorLabel.MOVEMENT)


class TestParseTs:
    def test_naive_uses_given_tz(self):
        # 2026-08-19 10:00:00 +08:00 == 02:00:00 UTC
        assert parse_ts("2026-08-19 10:00:00.000", SH) == 1787104800000

    def test_offset_in_string_wins_over_tz(self):
        a = parse_ts("2026-08-19T10:00:00+00:00", SH)
        b = parse_ts("2026-08-19 10:00:00", ZoneInfo("UTC"))
        assert a == b

    def test_accepts_epoch_millis(self):
        assert parse_ts("1787104800000", SH) == 1787104800000

    def test_seconds_precision_ok(self):
        assert parse_ts("2026-08-19 10:00:00", SH) == 1787104800000

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_ts("not-a-time", SH)


class TestMergeWindows:
    def _w(self, ts_ms, label, conf=0.9):
        return {"ts_ms": ts_ms, "label": label, "conf": conf}

    def test_merges_consecutive_same_label(self):
        wins = [self._w(i * 1000, SLEEP) for i in range(5)]
        ev = merge_windows(wins, window_sec=2.0, max_gap_sec=None)
        assert len(ev) == 1
        assert ev[0]["start_ms"] == 0
        assert ev[0]["end_ms"] == 4000 + 2000   # 末窗起点 + 窗长
        assert ev[0]["n_windows"] == 5

    def test_splits_on_label_change(self):
        wins = [self._w(i * 1000, SLEEP) for i in range(3)] + \
               [self._w((3 + i) * 1000, MOVE) for i in range(3)]
        ev = merge_windows(wins, 2.0, None)
        assert [e["label"] for e in ev] == [SLEEP, MOVE]

    def test_splits_on_time_gap(self):
        """录制中断两侧不该被合成一段连续行为。"""
        wins = [self._w(i * 1000, SLEEP) for i in range(3)]
        wins += [self._w(3_600_000 + i * 1000, SLEEP) for i in range(3)]
        ev = merge_windows(wins, 2.0, max_gap_sec=None)
        assert len(ev) == 2

    def test_explicit_max_gap_respected(self):
        wins = [self._w(0, SLEEP), self._w(10_000, SLEEP)]
        assert len(merge_windows(wins, 2.0, max_gap_sec=5)) == 2
        assert len(merge_windows(wins, 2.0, max_gap_sec=30)) == 1

    def test_confidence_is_mean(self):
        wins = [self._w(0, SLEEP, 0.8), self._w(1000, SLEEP, 1.0)]
        assert merge_windows(wins, 2.0, None)[0]["conf"] == pytest.approx(0.9)

    def test_unsorted_input_is_sorted(self):
        wins = [self._w(2000, SLEEP), self._w(0, SLEEP), self._w(1000, SLEEP)]
        ev = merge_windows(wins, 2.0, None)
        assert len(ev) == 1 and ev[0]["start_ms"] == 0

    def test_empty(self):
        assert merge_windows([], 2.0, None) == []


class TestDeviceMap:
    def _map(self, rows, tmp_path):
        p = tmp_path / "dm.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["match", "device_sn", "device_id",
                                              "bind_id", "timezone"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return DeviceMap.load(str(p))

    def test_substring_match(self, tmp_path):
        m = self._map([{"match": "IMU1", "device_sn": "AA", "device_id": "",
                        "bind_id": "", "timezone": ""}], tmp_path)
        assert m.resolve("rec_IMU1_infer.json")["device_sn"] == "AA"
        assert m.resolve("rec_IMU9_infer.json") is None

    def test_case_insensitive(self, tmp_path):
        m = self._map([{"match": "imu1", "device_sn": "AA", "device_id": "",
                        "bind_id": "", "timezone": ""}], tmp_path)
        assert m.resolve("REC_IMU1.csv")["device_sn"] == "AA"

    def test_longest_match_wins(self, tmp_path):
        """IMU1 不能把本该属于 IMU10 的文件抢走。"""
        m = self._map([
            {"match": "IMU1",  "device_sn": "A", "device_id": "", "bind_id": "", "timezone": ""},
            {"match": "IMU10", "device_sn": "B", "device_id": "", "bind_id": "", "timezone": ""},
        ], tmp_path)
        assert m.resolve("rec_IMU10.csv")["device_sn"] == "B"
        assert m.resolve("rec_IMU1.csv")["device_sn"] == "A"

    def test_requires_sn_or_id(self, tmp_path):
        with pytest.raises(SystemExit, match="至少填一个"):
            self._map([{"match": "IMU1", "device_sn": "", "device_id": "",
                        "bind_id": "", "timezone": ""}], tmp_path)

    def test_requires_match_column(self, tmp_path):
        with pytest.raises(SystemExit, match="match"):
            self._map([{"match": "", "device_sn": "AA", "device_id": "",
                        "bind_id": "", "timezone": ""}], tmp_path)


class TestReadInferJson:
    def test_parses_windows(self, tmp_path):
        p = tmp_path / "x_infer.json"
        p.write_text(json.dumps({"windows": [
            {"ts": "2026-08-19 10:00:00.000", "label": "睡觉", "conf": 0.98},
            {"ts": "2026-08-19 10:00:01.000", "label": "活动", "conf": 0.80},
        ]}, ensure_ascii=False), encoding="utf-8")
        rows = read_infer_json(p, SH)
        assert [r["label"] for r in rows] == [SLEEP, MOVE]
        assert rows[0]["conf"] == 0.98

    def test_skips_unknown_labels(self, tmp_path):
        p = tmp_path / "x_infer.json"
        p.write_text(json.dumps({"windows": [
            {"ts": "2026-08-19 10:00:00.000", "label": "吃饭", "conf": 0.9},
            {"ts": "2026-08-19 10:00:01.000", "label": "睡觉", "conf": 0.9},
        ]}, ensure_ascii=False), encoding="utf-8")
        assert len(read_infer_json(p, SH)) == 1


class TestReadCsvRows:
    def _write(self, tmp_path, text):
        p = tmp_path / "in.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_detects_window_format(self, tmp_path):
        p = self._write(tmp_path, "device_sn,ts,label,conf\n"
                                  "AA,2026-08-19 10:00:00.000,睡觉,0.9\n")
        kind, rows = read_csv_rows(p, SH)
        assert kind == "window" and rows[0]["label"] == SLEEP

    def test_detects_event_format(self, tmp_path):
        p = self._write(tmp_path, "device_sn,start_ts,end_ts,label,conf\n"
                                  "AA,2026-08-19 10:00:00,2026-08-19 10:05:00,活动,0.8\n")
        kind, rows = read_csv_rows(p, SH)
        assert kind == "event"
        assert rows[0]["end_ms"] - rows[0]["start_ms"] == 300_000

    def test_rejects_unknown_header(self, tmp_path):
        p = self._write(tmp_path, "foo,bar\n1,2\n")
        with pytest.raises(SystemExit, match="表头无法识别"):
            read_csv_rows(p, SH)
