import sys, os, json, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from zoneinfo import ZoneInfo

import pytest

from backfill.import_infer import (
    DeviceMap,
    collect_inputs,
    merge_windows,
    parse_ts,
    read_csv_rows,
    read_infer_json,
    smooth_windows,
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
        """ValueError (不是 SystemExit)，让调用方能跳过单个坏文件继续处理其它文件。"""
        p = self._write(tmp_path, "foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="表头无法识别"):
            read_csv_rows(p, SH)


# ---------------------------------------------------------------------------
# 时区归一（CST 等非 IANA 写法）
# ---------------------------------------------------------------------------

from datetime import datetime

from timezones import canonical_name, resolve


class TestTimezoneResolve:
    def _off(self, name):
        return datetime(2026, 8, 19, 12, tzinfo=resolve(name)).utcoffset().total_seconds() / 3600

    def test_cst_defaults_to_us_eastern(self):
        """ZoneInfo('CST') 会直接抛异常，必须走别名表，否则静默退回 UTC。

        生产库里 "CST" 用户与真实的 America/New_York 用户同批出现，
        经业务确认按美国东部时间解释（而不是字面上的中国标准时间）。
        """
        assert self._off("CST") == -4   # 8 月是 EDT
        assert canonical_name("CST") == "America/New_York"

    def test_iana_passthrough(self):
        assert canonical_name("Asia/Shanghai") == "Asia/Shanghai"
        assert self._off("Asia/Shanghai") == 8

    def test_common_aliases(self):
        for name in ("PRC", "Asia/Beijing", "Asia/Chongqing", "beijing", "China"):
            assert self._off(name) == 8, name

    def test_fixed_offset_forms(self):
        for name in ("+08:00", "UTC+8", "GMT+08", "utc+8"):
            assert self._off(name) == 8, name

    def test_negative_offset(self):
        assert self._off("-05:00") == -5

    def test_unknown_falls_back_to_utc(self):
        assert self._off("完全不存在的时区") == 0

    def test_empty_falls_back(self):
        assert self._off("") == 0
        assert self._off(None) == 0

    def test_non_china_iana_still_works(self):
        """别名表不能把其它地区的合法 IANA 名也拽到中国时区。"""
        assert self._off("America/Chicago") == -5   # 8月是 CDT


class TestJobsTimezoneHandling:
    def test_cst_local_str_matches_us_eastern(self):
        """修复前 'CST' 会退回 UTC；现在按业务确认的美国东部时间解释。"""
        from scheduler.jobs import _ts_to_local_str
        ts = 1787104800000  # 2026-08-19 10:00 +08:00 == 2026-08-18 22:00 EDT
        assert _ts_to_local_str(ts, "CST") == _ts_to_local_str(ts, "America/New_York")
        assert _ts_to_local_str(ts, "CST") != _ts_to_local_str(ts, "UTC")

    def test_cst_day_boundary_matches_us_eastern(self):
        """日聚合边界错位会直接污染每日汇总和皮肤评估。"""
        from scheduler.jobs import _day_start_utc_ms
        ts = 1787140800000  # 2026-08-19 20:00 +08:00
        assert _day_start_utc_ms(ts, "CST") == _day_start_utc_ms(ts, "America/New_York")

    def test_us_timezone_abbreviations(self):
        """生产库真实数据里出现过 EDT，必须映射到美国时区而不是静默退回 UTC。"""
        assert canonical_name("EDT") == "America/New_York"
        assert canonical_name("EST") == "America/New_York"
        assert canonical_name("PDT") == "America/Los_Angeles"
        assert canonical_name("PST") == "America/Los_Angeles"


class TestCollectInputs:
    def _make_tree(self, tmp_path):
        (tmp_path / "_infer").mkdir()
        (tmp_path / "_infer" / "a_infer.json").write_text("{}")
        (tmp_path / "by_conf_max" / "clips_0.5-0.6").mkdir(parents=True)
        (tmp_path / "by_conf_max" / "clips_0.5-0.6" / "clip_infer_result_majority.csv").write_text(
            "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,timestamp\n0,0,9.8,0,0,0,2026-08-19 08:00:00\n"
        )
        (tmp_path / "imu_daily_scratch_stats.csv").write_text("x\n1\n")
        return tmp_path

    def test_default_ignores_raw_clip_csvs(self, tmp_path):
        """默认只找 *_infer.json，run_review_bins_all_days.sh 输出目录里的
        复核用原始片段 CSV（acc_x/.../timestamp 表头，不是推理结果）不会被误收。"""
        root = self._make_tree(tmp_path)
        files = collect_inputs(str(root))
        assert [f.name for f in files] == ["a_infer.json"]

    def test_include_csv_pulls_in_csvs_too(self, tmp_path):
        root = self._make_tree(tmp_path)
        files = collect_inputs(str(root), include_csv=True)
        names = {f.name for f in files}
        assert "a_infer.json" in names
        assert "clip_infer_result_majority.csv" in names

    def test_excludes_daily_scratch_stats_even_with_include_csv(self, tmp_path):
        root = self._make_tree(tmp_path)
        files = collect_inputs(str(root), include_csv=True)
        assert "imu_daily_scratch_stats.csv" not in {f.name for f in files}

    def test_single_file_passthrough(self, tmp_path):
        f = tmp_path / "x_infer.json"
        f.write_text("{}")
        assert collect_inputs(str(f)) == [f]


class TestSmoothWindows:
    def _w(self, ts_ms, label, conf=0.9):
        return {"ts_ms": ts_ms, "label": label, "conf": conf}

    def test_removes_isolated_flip(self):
        """imu_train 的逐窗口原始预测没做平滑，单窗口误判会被切成几秒钟的
        碎片事件——这正是要修的问题。"""
        wins = ([self._w(i * 1000, SLEEP) for i in range(3)]
                + [self._w(3000, MOVE)]
                + [self._w(i * 1000, SLEEP) for i in range(4, 7)])
        out = smooth_windows(wins, smooth_k=5, window_sec=1.0)
        assert [w["label"] for w in out] == [SLEEP] * 7

    def test_k1_disables_smoothing(self):
        wins = [self._w(0, SLEEP), self._w(1000, MOVE), self._w(2000, SLEEP)]
        out = smooth_windows(wins, smooth_k=1, window_sec=1.0)
        assert out is wins

    def test_empty_input(self):
        assert smooth_windows([], smooth_k=5, window_sec=1.0) == []

    def test_does_not_smooth_across_recording_gap(self):
        """中断两侧是两段互不相关的行为，平滑不能把边界几帧强行拉成同一类。"""
        wins = ([self._w(i * 1000, SLEEP) for i in range(3)]
                + [self._w(3_600_000 + i * 1000, MOVE) for i in range(3)])
        out = smooth_windows(wins, smooth_k=5, window_sec=1.0)
        labels = [w["label"] for w in out]
        assert labels == [SLEEP, SLEEP, SLEEP, MOVE, MOVE, MOVE]

    def test_reduces_fragment_event_count(self):
        """平滑前后拿去合并事件，事件数应该明显变少——这是它存在的意义。"""
        rng_labels = [SLEEP, SLEEP, MOVE, SLEEP, SLEEP, SLEEP, MOVE, SLEEP,
                     SLEEP, SLEEP, SLEEP, MOVE, SLEEP, SLEEP]
        wins = [self._w(i * 1000, lbl) for i, lbl in enumerate(rng_labels)]
        raw_events = merge_windows(wins, 1.0, None)
        smoothed = smooth_windows(wins, smooth_k=5, window_sec=1.0)
        smoothed_events = merge_windows(smoothed, 1.0, None)
        assert len(smoothed_events) < len(raw_events)

    def test_smoothing_preserves_other_fields(self):
        wins = [self._w(i * 1000, SLEEP, conf=0.5 + i * 0.1) for i in range(5)]
        out = smooth_windows(wins, smooth_k=3, window_sec=1.0)
        assert [w["conf"] for w in out] == [w["conf"] for w in wins]
        assert [w["ts_ms"] for w in out] == [w["ts_ms"] for w in wins]
