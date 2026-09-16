"""佩戴时长不能把「未佩戴」算进去。

    python -m pytest tests/unit/test_wear_minutes.py -q

这条 SQL 以前是"把当天所有行为事件的时长加起来"。只有 3 类的时候没问题——
每个事件都意味着项圈戴在身上。换成 5 类模型之后，摘下项圈那段会产生
behavior=4 的事件，照旧全加的话**「没戴」的时间被算成了「戴着」**，方向正好反。

一处错、四处连带，而且全都不报错：

  · wear_minutes 虚高 → 该标无效天（data_quality=1）的没标
  · sleep_ratio / active_ratio 的分母偏大 → 比值偏低 → 状态灯判错
  · off_min = 1440 - wear - loose → 偏小，而真正的未佩戴时长就在 behavior=4 里
  · sleep_min + move_min + scratch_min 不再约等于 wear_min —— 多出来那块
    没有任何一列装它，后端对账时会发现"对不上"

这里**真的建一张表、真的执行那条 SQL**（sqlite），不是比对字符串：
比字符串的话，把 `<>` 写成 `=` 照样能通过。

SQL 放在 modules/assessment/queries.py（不 import 任何重依赖），
所以这个文件不用为了跑几条 assert 去打 loguru / joblib / aiomysql 的桩。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.inference.labels import BehaviorLabel  # noqa: E402


from modules.assessment.queries import NOT_WEARING, wear_ms_sql  # noqa: E402

DAY = 86_400_000
MIN = 60_000


@pytest.fixture
def db():
    eng = create_engine("sqlite://")
    with eng.connect() as c:
        c.execute(text("""
            CREATE TABLE beh (
                ts_start BIGINT, ts_end BIGINT, behavior SMALLINT
            )
        """))
        yield c


def _add(db, behavior, minutes, start_min=0):
    db.execute(text("INSERT INTO beh VALUES (:a, :b, :c)"), {
        "a": start_min * MIN, "b": (start_min + minutes) * MIN, "c": int(behavior)})


def _wear_min(db):
    ms = db.execute(text(wear_ms_sql("beh")), {
        "day_start": 0, "day_end": DAY,
        "not_worn": NOT_WEARING,
    }).scalar() or 0
    return int(ms) // MIN


def test_not_worn_is_excluded(db):
    """核心那条。戴着 600 分钟 + 没戴 300 分钟 → 佩戴时长是 600。"""
    _add(db, BehaviorLabel.SLEEP, 400, 0)
    _add(db, BehaviorLabel.MOVEMENT, 200, 400)
    _add(db, BehaviorLabel.NOT_WORN, 300, 600)
    assert _wear_min(db) == 600


def test_worn_classes_all_count(db):
    """睡觉 / 活动 / 抓挠 / 未知 都算佩戴。

    未知（0）也要算：它是"模型不确定"，不是"没戴"——3 类时代
    confidence_threshold 产生的就是它，那些时段项圈是戴着的。
    """
    for lab in (BehaviorLabel.UNKNOWN, BehaviorLabel.MOVEMENT,
                BehaviorLabel.SLEEP, BehaviorLabel.SCRATCH):
        _add(db, lab, 10)
    assert _wear_min(db) == 40


def test_three_class_data_is_unaffected(db):
    """老模型不产生 behavior=4，所以这个条件对老数据是空操作。

    要是它顺手把别的类别也排掉了，历史那些天的 wear_min 会整体变小，
    而 data_quality / 状态灯跟着一起变——那是一次静默的历史数据改写。
    """
    _add(db, BehaviorLabel.SLEEP, 500)
    _add(db, BehaviorLabel.MOVEMENT, 300)
    _add(db, BehaviorLabel.SCRATCH, 5)
    assert _wear_min(db) == 805


def test_a_full_day_of_not_worn_is_zero_wear(db):
    """项圈一整天没戴 → 佩戴 0 分钟 → data_quality 会判成无效天。

    修之前这里会返回 1440，也就是"戴了一整天"，然后这一天的睡眠比、
    活动比、状态灯全部照常算出来——**看数据完全正常**。
    """
    _add(db, BehaviorLabel.NOT_WORN, 1440)
    assert _wear_min(db) == 0


def test_events_outside_the_day_are_excluded(db):
    """日期边界没被这次改动碰到，顺带钉住。"""
    _add(db, BehaviorLabel.SLEEP, 60, 0)
    db.execute(text("INSERT INTO beh VALUES (:a, :b, :c)"), {
        "a": DAY + MIN, "b": DAY + 61 * MIN, "c": int(BehaviorLabel.SLEEP)})
    assert _wear_min(db) == 60


def test_summary_columns_add_up_to_wear(db):
    """sleep + move + scratch ≈ wear_min —— 后端对账靠的就是这个。

    未佩戴被算进 wear 的话这个等式破了，而每一列单独看都正常。
    """
    _add(db, BehaviorLabel.SLEEP, 500, 0)
    _add(db, BehaviorLabel.MOVEMENT, 300, 500)
    _add(db, BehaviorLabel.SCRATCH, 5, 800)
    _add(db, BehaviorLabel.NOT_WORN, 400, 805)
    assert _wear_min(db) == 500 + 300 + 5
