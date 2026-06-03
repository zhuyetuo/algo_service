import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import calendar
import datetime
from unittest.mock import MagicMock, patch

import pytest

from db.tdengine import _ts_to_ms


class TestTsToMs:
    def test_int_passthrough(self):
        assert _ts_to_ms(1_000_000) == 1_000_000

    def test_string_conversion(self):
        assert _ts_to_ms("9876543") == 9876543

    def test_datetime_conversion(self):
        dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
        expected = calendar.timegm(dt.utctimetuple()) * 1000
        assert _ts_to_ms(dt) == expected

    def test_datetime_with_microseconds(self):
        dt = datetime.datetime(2024, 1, 1, 0, 0, 0, microsecond=500_000)
        result = _ts_to_ms(dt)
        base_ms = calendar.timegm(dt.utctimetuple()) * 1000
        expected = base_ms + (500_000 // 1000)
        assert result == expected


class TestTdGetDevices:
    def test_td_get_devices_returns_list(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("device_001",), (" device_002 ",)]
        mock_conn.cursor.return_value = mock_cursor

        with patch("taosrest.connect", return_value=mock_conn):
            from db.tdengine import td_get_devices
            devices = td_get_devices()

        assert "device_001" in devices
        assert "device_002" in devices  # stripped
        mock_conn.close.assert_called_once()


class TestTdFetch:
    def test_td_fetch_returns_rows(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1704067200000, 0.1, 0.2, 9.8, 0.01, 0.02, 0.03),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("taosrest.connect", return_value=mock_conn):
            from db.tdengine import td_fetch
            rows = td_fetch("device_001", 0)

        assert len(rows) == 1
        assert rows[0]["ts_ms"] == 1704067200000
        assert rows[0]["ax"] == pytest.approx(0.1)
        mock_conn.close.assert_called_once()

    def test_td_fetch_empty(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        with patch("taosrest.connect", return_value=mock_conn):
            from db.tdengine import td_fetch
            rows = td_fetch("device_001", 0)

        assert rows == []
