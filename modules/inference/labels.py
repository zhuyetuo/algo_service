"""行为标签的编码表。

单独一个模块、**不 import 任何重依赖**（joblib / sklearn / numpy 都没有），
这样装模型的脚本、写库那边、测试都能直接拿到它，不用为了查一个映射表
把整个推理模块和它的依赖全拖起来。

类别中文名 → 编码这件事只能有一个地方定义。散在两处的话，加一个类别时
漏改一处的表现是：那些窗口识别出来了，但写不进库，被静默丢掉。
"""

from enum import IntEnum


class BehaviorLabel(IntEnum):
    UNKNOWN  = 0
    MOVEMENT = 1
    SLEEP    = 2
    SCRATCH  = 3


# imu_train 类别中文名 → BehaviorLabel
# 类别顺序由 ml_rf.json 的 classes 决定，不写死下标——换模型时顺序可能变
ZH_TO_LABEL: dict[str, int] = {
    "抓挠": int(BehaviorLabel.SCRATCH),
    "活动": int(BehaviorLabel.MOVEMENT),
    "睡觉": int(BehaviorLabel.SLEEP),
}

LABEL_ZH: dict[int, str] = {
    int(BehaviorLabel.UNKNOWN):  "未知",
    int(BehaviorLabel.MOVEMENT): "活动",
    int(BehaviorLabel.SLEEP):    "睡觉",
    int(BehaviorLabel.SCRATCH):  "抓挠",
}

# classes 缺失时的兜底顺序（与仓库内已提交模型一致）
DEFAULT_CLASSES = ["抓挠", "活动", "睡觉"]
