"""评估用到的几条 SQL，单独放这里。

**不 import 任何重依赖**（loguru / joblib / sklearn / aiomysql 都没有），
跟 labels.py 一个道理：这些 SQL 里藏着业务规则，规则得能被直接测，
而不用为了跑一条 assert 把整个评估流程和它的依赖全拖起来。

evaluator.py 那边把它们包成 text(...) 再执行。
"""

from modules.inference.labels import BehaviorLabel


def wear_ms_sql(b_tbl: str) -> str:
    """当天"项圈戴在身上"的总毫秒数。**排掉未佩戴**（behavior=4）。

    参数：:day_start / :day_end / :not_worn

    这一段以前是"把当天所有行为事件的时长加起来"。只有 3 类的时候没问题——
    每个事件都意味着项圈戴在身上。换成 5 类模型之后，摘下项圈那段会产生
    behavior=4 的事件，照旧全加的话**「没戴」的时间被算成了「戴着」**，
    方向正好反了。

    一处错、四处连带，而且一个报错都没有：
      · wear_minutes 虚高 → 该标无效天（data_quality=1）的没标
      · sleep_ratio / active_ratio 的分母偏大 → 比值偏低 → 状态灯判错
      · off_min = 1440 - wear - loose → 偏小，而真正的未佩戴时长
        明明就在 behavior=4 里
      · sleep_min + move_min + scratch_min 不再约等于 wear_min，
        因为多出来那块没有任何一列装它——后端对账时就会发现"对不上"

    3 类模型不产生 behavior=4，所以这个条件对老数据是空操作，
    不用按模型版本分支。
    """
    return f"""
        SELECT COALESCE(SUM(ts_end - ts_start), 0) AS wore_ms
        FROM   {b_tbl}
        WHERE  ts_start >= :day_start
          AND  ts_start  < :day_end
          AND  behavior <> :not_worn
    """


#: 算佩戴时长时要排掉的行为编码。只有"没戴"这一种——
#: "未知"是模型不确定，那时候项圈是戴着的，不能排。
NOT_WEARING = int(BehaviorLabel.NOT_WORN)
