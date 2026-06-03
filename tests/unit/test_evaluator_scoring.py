import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from modules.assessment.evaluator import (
    eval_phase,
    get_threshold,
    _calc_s1,
    _calc_s2,
    _calc_s3,
    _calc_s4,
    _calc_s6,
    _score_to_level,
    _get_alert_level,
)


class TestEvalPhase:
    @pytest.mark.parametrize("vd,expected", [
        (0, 0), (1, 0), (2, 0),
        (3, 1), (7, 1), (13, 1),
        (14, 2), (20, 2), (29, 2),
        (30, 3), (60, 3),
    ])
    def test_eval_phase(self, vd, expected):
        assert eval_phase(vd) == expected


class TestGetThreshold:
    def test_warmup_returns_none(self):
        for vd in (0, 1, 2):
            assert get_threshold(vd) is None

    def test_phase1(self):
        t = get_threshold(3)
        assert t is not None
        assert t.z == pytest.approx(4.0)
        assert t.consec == 5
        assert t.avg_z == pytest.approx(5.0)

    def test_phase2(self):
        t = get_threshold(14)
        assert t is not None
        assert t.z == pytest.approx(3.5)
        assert t.consec == 4
        assert t.avg_z == pytest.approx(4.0)

    def test_phase3(self):
        t = get_threshold(30)
        assert t is not None
        assert t.z == pytest.approx(2.5)
        assert t.consec == 3
        assert t.avg_z == pytest.approx(3.5)


class TestCalcS1:
    def test_zero_or_negative(self):
        assert _calc_s1(0.0) == pytest.approx(0.0)
        assert _calc_s1(-1.0) == pytest.approx(0.0)
        assert _calc_s1(-100.0) == pytest.approx(0.0)

    def test_linear_midpoint(self):
        # z=2.0 -> 2/4*25 = 12.5
        assert _calc_s1(2.0) == pytest.approx(12.5)

    def test_caps_at_25(self):
        assert _calc_s1(4.0) == pytest.approx(25.0)
        assert _calc_s1(10.0) == pytest.approx(25.0)


class TestCalcS2:
    def test_at_baseline(self):
        # wpeb=mean -> z=0 -> 0
        assert _calc_s2(5.0, 5.0, 1.0) == pytest.approx(0.0)

    def test_below_baseline(self):
        # wpeb < mean -> z < 0 -> 0
        assert _calc_s2(3.0, 5.0, 1.0) == pytest.approx(0.0)

    def test_high_above(self):
        # wpeb=9, mean=5, std=1 -> z=4 -> 25
        assert _calc_s2(9.0, 5.0, 1.0) == pytest.approx(25.0)

    def test_std_floor(self):
        # std=0 should not crash; floor is 1.0, so wpeb=6, mean=5 -> z=1 -> 6.25
        result = _calc_s2(6.0, 5.0, 0.0)
        assert result >= 0.0


class TestCalcS3:
    @pytest.mark.parametrize("scratch,sleep,expected", [
        (0, 0, 0),
        (1, 1, 5),
        (2, 2, 8),
        (3, 4, 17),
        (6, 4, 20),
    ])
    def test_calc_s3(self, scratch, sleep, expected):
        assert _calc_s3(scratch, sleep) == pytest.approx(float(expected))


class TestCalcS4:
    def test_none_temp(self):
        assert _calc_s4(None, 0.0) == pytest.approx(0.0)

    def test_high_loose(self):
        # loose_minutes > 240 -> 0
        assert _calc_s4(39.0, 241.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("temp,expected", [
        (37.9, 0.0),
        (38.5, 2.0),
        (39.3, 6.0),
        (40.0, 10.0),
    ])
    def test_temp_ranges(self, temp, expected):
        assert _calc_s4(temp, 0.0) == pytest.approx(expected)


class TestCalcS6:
    def test_no_data(self):
        assert _calc_s6(None, None) == pytest.approx(0.0)

    def test_high_temp(self):
        assert _calc_s6(35.0, None) == pytest.approx(-2.0)

    def test_low_temp(self):
        assert _calc_s6(5.0, None) == pytest.approx(1.0)

    def test_low_humidity(self):
        assert _calc_s6(None, 30.0) == pytest.approx(2.0)

    def test_high_humidity(self):
        assert _calc_s6(None, 85.0) == pytest.approx(-1.0)

    def test_cancels_out(self):
        # high temp (-2) + low humidity (+2) = 0
        assert _calc_s6(35.0, 30.0) == pytest.approx(0.0)

    def test_combined_positive(self):
        # low temp (+1) + low humidity (+2) = 3
        assert _calc_s6(5.0, 30.0) == pytest.approx(3.0)

    def test_combined_negative(self):
        # high temp (-2) + high humidity (-1) = -3
        assert _calc_s6(35.0, 85.0) == pytest.approx(-3.0)


class TestScoreToLevel:
    @pytest.mark.parametrize("score,expected", [
        (0, 0),
        (9.9, 0),
        (10, 1),
        (50, 5),
        (99, 9),
        (100, 10),
    ])
    def test_score_to_level(self, score, expected):
        assert _score_to_level(score) == expected


class TestGetAlertLevel:
    @pytest.mark.parametrize("consec,expected", [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (10, 3),
    ])
    def test_get_alert_level(self, consec, expected):
        assert _get_alert_level(consec) == expected
