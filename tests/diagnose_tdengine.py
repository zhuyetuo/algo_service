"""
TDengine 连接诊断脚本。

用法：
    # 宿主机直接运行（用默认配置）
    python tests/diagnose_tdengine.py

    # 指定连接参数
    TD_HOST=192.168.1.10 TD_PORT=6041 python tests/diagnose_tdengine.py

    # 容器内运行
    docker exec -it algo_service python tests/diagnose_tdengine.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import taosrest
from config import settings

SEP = "─" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg):  print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}")
def info(msg): print(f"     {msg}")


def main():
    print(f"\n{'='*60}")
    print("  TDengine 连接诊断")
    print(f"{'='*60}")
    info(f"host     : {settings.td_host}:{settings.td_port}")
    info(f"user     : {settings.td_user}")
    info(f"database : {settings.td_database}")
    info(f"supertable: {settings.td_supertable}")

    # ── 1. 建立连接 ──────────────────────────────────────────────
    section("1. 连接测试")
    try:
        conn = taosrest.connect(
            url=f"http://{settings.td_host}:{settings.td_port}",
            user=settings.td_user,
            password=settings.td_password,
        )
        cursor = conn.cursor()
        ok(f"连接成功 → http://{settings.td_host}:{settings.td_port}")
    except Exception as e:
        fail(f"连接失败: {e}")
        sys.exit(1)

    # ── 2. 列出所有数据库 ─────────────────────────────────────────
    section("2. 数据库列表")
    try:
        cursor.execute("SHOW DATABASES")
        dbs = [r[0] for r in cursor.fetchall()]
        for db in dbs:
            marker = " ← 当前配置" if db == settings.td_database else ""
            info(f"  {db}{marker}")
        if settings.td_database in dbs:
            ok(f"目标数据库 '{settings.td_database}' 存在")
        else:
            fail(f"目标数据库 '{settings.td_database}' 不存在！可用: {dbs}")
    except Exception as e:
        fail(f"SHOW DATABASES 失败: {e}")

    # ── 3. 列出超级表 ─────────────────────────────────────────────
    section(f"3. 超级表列表（{settings.td_database}）")
    try:
        cursor.execute(f"SHOW {settings.td_database}.STABLES")
        stables = [r[0] for r in cursor.fetchall()]
        for st in stables:
            marker = " ← 当前配置" if st == settings.td_supertable else ""
            info(f"  {st}{marker}")
        if settings.td_supertable in stables:
            ok(f"目标超级表 '{settings.td_supertable}' 存在")
        else:
            fail(f"目标超级表 '{settings.td_supertable}' 不存在！可用: {stables}")
    except Exception as e:
        fail(f"SHOW STABLES 失败: {e}")

    # ── 4. 查询设备列表 ───────────────────────────────────────────
    section(f"4. 设备列表（DISTINCT device_sn）")
    full_table = f"{settings.td_database}.{settings.td_supertable}"
    try:
        cursor.execute(f"SELECT DISTINCT device_sn FROM {full_table}")
        devices = [str(r[0]).strip() for r in cursor.fetchall()]
        if devices:
            ok(f"共 {len(devices)} 台设备")
            for d in devices[:10]:
                info(f"  {d}")
            if len(devices) > 10:
                info(f"  ... 共 {len(devices)} 台")
        else:
            fail("未找到任何设备（表存在但无数据）")
    except Exception as e:
        fail(f"查询设备列表失败: {e}")

    # ── 5. 抽样数据 ───────────────────────────────────────────────
    section(f"5. 抽样数据（最新 3 行）")
    try:
        cursor.execute(f"SELECT ts, ax, ay, az, gx, gy, gz, device_sn FROM {full_table} ORDER BY ts DESC LIMIT 3")
        rows = cursor.fetchall()
        if rows:
            ok(f"取到 {len(rows)} 行样本数据")
            for r in rows:
                info(f"  ts={r[0]}  ax={r[1]:.3f}  device_sn={r[7]}")
        else:
            fail("表存在但无数据")
    except Exception as e:
        fail(f"抽样查询失败: {e}")

    conn.close()
    print(f"\n{'='*60}\n  诊断完成\n{'='*60}\n")


if __name__ == "__main__":
    main()
