#!/bin/bash
# 清空算法服务所有 MySQL 数据（行为、环境、评估、基线、汇总、同步断点）
# 用法: bash scripts/reset_db.sh

DB_HOST="${DB_HOST:-192.168.33.253}"
DB_PORT="${DB_PORT:-30100}"
DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-Hicc-pet-mysql-2026}"

MYSQL="docker exec local-mysql8 mysql -h ${DB_HOST} -P ${DB_PORT} -u ${DB_USER} -p${DB_PASSWORD}"
DEVICES=(70 72 73 94 95 118)

echo ">>> 开始清空算法服务数据库..."

# 按设备删除各分表
for id in "${DEVICES[@]}"; do
  $MYSQL 2>/dev/null -e "
    DROP TABLE IF EXISTS pet_dog_behavior.d_${id};
    DROP TABLE IF EXISTS pet_dog_environment.d_${id};
    DROP TABLE IF EXISTS pet_dog_daily_summary.d_${id};
    DROP TABLE IF EXISTS pet_dog_skin_assessment.d_${id};
    DROP TABLE IF EXISTS pet_dog_wear_event.d_${id};
  "
  echo "    设备 ${id} 分表已删除"
done

# 公共表
$MYSQL 2>/dev/null -e "
  DROP TABLE IF EXISTS pet_dog_scratch_baseline.pet_skin_baseline;
  DROP TABLE IF EXISTS algo.processing_errors;
  DROP TABLE IF EXISTS algo.device_sync_state;
"

echo ">>> 完成，重启服务让各表自动重建..."
docker restart algo_service
echo ">>> 服务已重启"
