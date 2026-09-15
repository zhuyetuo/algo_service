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
    # 5 类模型才有。项圈没戴在身上的时段。
    #
    # 为什么单独一个编码、不并进"未知"：以前没有这个类别时，摘下项圈那段
    # 会被模型判成睡觉（三个类别里必须挑一个），于是 sleep_min 虚高，
    # 而看数据完全看不出来——跟真的在睡长得一模一样。
    # 并进"未知"的话"不知道"和"项圈没戴"又分不开了。
    NOT_WORN = 4


# imu_train 类别中文名 → BehaviorLabel
# 类别顺序由 ml_rf.json 的 classes 决定，不写死下标——换模型时顺序可能变
ZH_TO_LABEL: dict[str, int] = {
    "抓挠": int(BehaviorLabel.SCRATCH),
    "活动": int(BehaviorLabel.MOVEMENT),
    "睡觉": int(BehaviorLabel.SLEEP),
    "未佩戴": int(BehaviorLabel.NOT_WORN),
    # **甩身体并进活动**：业务上不单独统计它，不值得为它开一个编码，
    # 而丢掉的话那些时段会变成空洞。
    #
    # 注意这只影响**写进库的编码**。后处理那一层仍然按"甩身体"这个类别
    # 参与计算（抓挠会吞并前后的甩身体窗口），所以这里不能在更早的地方
    # 就把它改名——那会让后处理少一条规则，抓挠段短一截。
    "甩身体": int(BehaviorLabel.MOVEMENT),
}

LABEL_ZH: dict[int, str] = {
    int(BehaviorLabel.UNKNOWN):  "未知",
    int(BehaviorLabel.MOVEMENT): "活动",
    int(BehaviorLabel.SLEEP):    "睡觉",
    int(BehaviorLabel.SCRATCH):  "抓挠",
    int(BehaviorLabel.NOT_WORN): "未佩戴",
}

# classes 缺失时的兜底顺序（与仓库内已提交模型一致）
DEFAULT_CLASSES = ["抓挠", "活动", "睡觉"]
