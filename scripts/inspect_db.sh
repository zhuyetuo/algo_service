#!/bin/bash
# 只读导出 MySQL 现有结构与样本数据，用于核对新写入的数据是否与现状一致。
#
# 用法:
#   bash scripts/inspect_db.sh                 # 直接用本机 mysql 客户端
#   MYSQL_VIA_DOCKER=1 bash scripts/inspect_db.sh   # 走 docker exec local-mysql8
#   bash scripts/inspect_db.sh > db_snapshot.txt 2>&1
#
# 全部是 SHOW / SELECT，不做任何写操作。

set -uo pipefail

DB_HOST="${DB_HOST:-192.168.33.253}"
DB_PORT="${DB_PORT:-30100}"
DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-Hicc-pet-mysql-2026}"
BIZ_SCHEMA="${BIZ_SCHEMA:-hiccpet_petos}"
DOCKER_MYSQL_CONTAINER="${DOCKER_MYSQL_CONTAINER:-local-mysql8}"

# 选择 mysql 客户端：显式指定 > 本机 mysql > docker exec
pick_client() {
  if [[ "${MYSQL_VIA_DOCKER:-0}" == "1" ]]; then
    MYSQL_CMD=(docker exec -i "$DOCKER_MYSQL_CONTAINER" mysql
               -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "-p${DB_PASSWORD}")
    CLIENT_DESC="docker exec ${DOCKER_MYSQL_CONTAINER} mysql"
    return
  fi
  if command -v mysql >/dev/null 2>&1; then
    MYSQL_CMD=(mysql --default-character-set=utf8mb4
               -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "-p${DB_PASSWORD}")
    CLIENT_DESC="本机 mysql 客户端"
    return
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$DOCKER_MYSQL_CONTAINER"; then
    MYSQL_CMD=(docker exec -i "$DOCKER_MYSQL_CONTAINER" mysql --default-character-set=utf8mb4
               -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "-p${DB_PASSWORD}")
    CLIENT_DESC="本机没有 mysql 客户端，自动改用 docker exec ${DOCKER_MYSQL_CONTAINER}"
    return
  fi
  echo "❌ 找不到可用的 mysql 客户端：" >&2
  echo "   - 本机没装 mysql（command -v mysql 为空）" >&2
  echo "   - 也没有名为 ${DOCKER_MYSQL_CONTAINER} 的运行中容器" >&2
  echo >&2
  echo "   解决办法（任选其一）：" >&2
  echo "     sudo apt install -y mysql-client" >&2
  echo "     DOCKER_MYSQL_CONTAINER=<你的mysql容器名> bash scripts/inspect_db.sh" >&2
  echo "     docker run --rm -i mysql:8 mysql -h ... （临时容器）" >&2
  exit 1
}
pick_client

# 注意：这里**不吞掉 stderr**。之前吞掉导致连接失败时全部小节静默为空，
# 看起来像"库里没数据"，实际是根本没连上。
q()  { "${MYSQL_CMD[@]}" --table    -e "$1"; }
qv() { "${MYSQL_CMD[@]}" --vertical -e "$1"; }
# 只在"表可能不存在"的探测处用这个，失败不打断整体流程
qq() { "${MYSQL_CMD[@]}" --table -e "$1" 2>&1 | grep -v "Using a password" || true; }

hr() { echo; echo "════════════════════════════════════════════════════════════════"; \
       echo "▶ $1"; echo "════════════════════════════════════════════════════════════════"; }

echo "MySQL 结构快照  $(date '+%Y-%m-%d %H:%M:%S')"
echo "目标: ${DB_HOST}:${DB_PORT}  用户: ${DB_USER}  业务库: ${BIZ_SCHEMA}"
echo "客户端: ${CLIENT_DESC}"

hr "0. 连通性与版本"
# 先硬探一次；连不上就直接退出，不再让后面几十条查询各自静默失败
if ! q "SELECT VERSION() AS mysql_version, NOW() AS db_now,
        @@system_time_zone AS sys_tz, @@time_zone AS session_tz;"; then
  echo >&2
  echo "❌ 连接失败，后面的查询就不跑了。排查方向：" >&2
  echo "   1) 端口/地址对不对：nc -vz ${DB_HOST} ${DB_PORT}" >&2
  echo "   2) 账号密码对不对（当前用 DB_USER=${DB_USER}）" >&2
  echo "   3) 该账号是否允许从这台机器连过来（MySQL 的 host 授权）" >&2
  echo "   4) 走容器试试：MYSQL_VIA_DOCKER=1 bash scripts/inspect_db.sh" >&2
  echo >&2
  echo "   .env 里的实际值：" >&2
  grep -E '^DB_(HOST|PORT|USER|PASSWORD)=' .env 2>/dev/null | sed 's/PASSWORD=.*/PASSWORD=***/' >&2
  exit 1
fi

hr "1. 所有库"
q "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME AS charset, DEFAULT_COLLATION_NAME AS collation
   FROM information_schema.SCHEMATA
   WHERE SCHEMA_NAME NOT IN ('information_schema','performance_schema','mysql','sys')
   ORDER BY SCHEMA_NAME;"

hr "2. 算法相关库的所有表（行数 + 字符集）"
q "SELECT TABLE_SCHEMA, TABLE_NAME, ENGINE, TABLE_ROWS, TABLE_COLLATION
   FROM information_schema.TABLES
   WHERE TABLE_SCHEMA IN ('algo','pet_dog_behavior','pet_dog_skin_assessment',
                          'pet_dog_environment','pet_dog_daily_summary',
                          'pet_dog_scratch_baseline','pet_dog_wear_event')
   ORDER BY TABLE_SCHEMA, TABLE_NAME;"

hr "3. 行为表结构（重点：新数据要跟这个完全一致）"
for t in $(q "SELECT TABLE_NAME FROM information_schema.TABLES
              WHERE TABLE_SCHEMA='pet_dog_behavior' ORDER BY TABLE_NAME LIMIT 3;" \
           | grep -oE 'd_[0-9]+'); do
  echo; echo "--- pet_dog_behavior.$t ---"
  qq "SHOW CREATE TABLE pet_dog_behavior.$t;"
  echo "-- 列定义 --"
  q "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY, EXTRA
     FROM information_schema.COLUMNS
     WHERE TABLE_SCHEMA='pet_dog_behavior' AND TABLE_NAME='$t' ORDER BY ORDINAL_POSITION;"
  echo "-- 索引 --"
  q "SHOW INDEX FROM pet_dog_behavior.$t;"
  echo "-- 最新 5 行 --"
  q "SELECT * FROM pet_dog_behavior.$t ORDER BY ts_start DESC LIMIT 5;"
  echo "-- behavior 取值分布 --"
  q "SELECT behavior, behavior_label, COUNT(*) AS n,
            MIN(confidence) AS conf_min, MAX(confidence) AS conf_max,
            MIN(duration_sec) AS dur_min, MAX(duration_sec) AS dur_max
     FROM pet_dog_behavior.$t GROUP BY behavior, behavior_label ORDER BY behavior;"
  echo "-- 时间范围 --"
  q "SELECT MIN(ts_start) AS ts_min, MAX(ts_start) AS ts_max,
            MIN(local_start) AS local_min, MAX(local_start) AS local_max,
            COUNT(DISTINCT user_timezone) AS tz_kinds,
            GROUP_CONCAT(DISTINCT user_timezone) AS timezones,
            COUNT(DISTINCT bind_id) AS bind_ids
     FROM pet_dog_behavior.$t;"
done

hr "4. 其它算法表结构"
for tbl in pet_dog_skin_assessment pet_dog_environment pet_dog_daily_summary; do
  t=$(q "SELECT TABLE_NAME FROM information_schema.TABLES
         WHERE TABLE_SCHEMA='$tbl' ORDER BY TABLE_NAME LIMIT 1;" | grep -oE 'd_[0-9]+' | head -1)
  [[ -z "$t" ]] && { echo; echo "--- $tbl 下没有分表 ---"; continue; }
  echo; echo "--- $tbl.$t ---"
  qq "SHOW CREATE TABLE $tbl.$t;"
  echo "-- 最新 3 行 --"
  qq "SELECT * FROM $tbl.$t LIMIT 3;"
done

echo; echo "--- pet_dog_scratch_baseline.pet_skin_baseline ---"
qq "SHOW CREATE TABLE pet_dog_scratch_baseline.pet_skin_baseline;"
qq "SELECT * FROM pet_dog_scratch_baseline.pet_skin_baseline LIMIT 5;"

hr "5. algo 库：同步断点与错误表"
qq "SHOW CREATE TABLE algo.device_sync_state;"
qq "SELECT * FROM algo.device_sync_state ORDER BY device_id;"
echo "-- processing_errors --"
qq "SHOW CREATE TABLE algo.processing_errors;"
qq "SELECT status, COUNT(*) AS n FROM algo.processing_errors GROUP BY status;"

hr "6. 业务库：设备与绑定（决定 device_id / bind_id / 时区怎么填）"
q "SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.TABLES
   WHERE TABLE_SCHEMA='${BIZ_SCHEMA}'
     AND TABLE_NAME IN ('device','device_bind_history','user','pet')
   ORDER BY TABLE_NAME;"

echo; echo "--- ${BIZ_SCHEMA}.device 列 ---"
q "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY
   FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA='${BIZ_SCHEMA}' AND TABLE_NAME='device' ORDER BY ORDINAL_POSITION;"
q "SELECT id, device_sn FROM ${BIZ_SCHEMA}.device ORDER BY id LIMIT 30;"

echo; echo "--- ${BIZ_SCHEMA}.device_bind_history 列 ---"
q "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY
   FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA='${BIZ_SCHEMA}' AND TABLE_NAME='device_bind_history' ORDER BY ORDINAL_POSITION;"
q "SELECT bind_id, device_id, user_id, pet_id, bind_status
   FROM ${BIZ_SCHEMA}.device_bind_history ORDER BY bind_id LIMIT 30;"

echo; echo "--- 活跃绑定 JOIN（推理周期实际用的那条查询）---"
q "SELECT dbh.bind_id, dbh.device_id, d.device_sn, dbh.user_id,
          COALESCE(u.timezone,'(NULL)') AS timezone
   FROM ${BIZ_SCHEMA}.device_bind_history dbh
   JOIN ${BIZ_SCHEMA}.device d ON dbh.device_id = d.id
   JOIN ${BIZ_SCHEMA}.\`user\` u ON dbh.user_id = u.id
   WHERE dbh.bind_status = 1
   ORDER BY dbh.device_id;"

hr "7. user.timezone 实际存了什么（CST 那个问题的关键）"
q "SELECT COALESCE(timezone,'(NULL)') AS timezone, COUNT(*) AS n
   FROM ${BIZ_SCHEMA}.\`user\` GROUP BY timezone ORDER BY n DESC;"

hr "8. 是否存在视图（后端可能读的是视图而不是表）"
q "SELECT TABLE_SCHEMA, TABLE_NAME FROM information_schema.VIEWS
   WHERE TABLE_SCHEMA LIKE 'pet_dog%' OR TABLE_SCHEMA='algo';"

echo
echo "════════════════════════════════════════════════════════════════"
echo "导出完成。把上面全部内容贴回来即可。"
echo "════════════════════════════════════════════════════════════════"
