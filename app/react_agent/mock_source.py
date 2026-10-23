"""
Mock 数据源（仅用于演示）。

=====================================================================
【模拟数据源】以下设备状态 / 用户权限查询均为本地 JSON 模拟数据，
生产环境需替换为真实 CMDB（设备监控）与 IAM / ITSM（权限、驱动）接口。
本模块只负责读取 JSON，不伪装成真实系统。
=====================================================================
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

from app.core.logger import logger

_MOCK_DIR = Path(__file__).resolve().parent / "mock_data"


def _load_json(file_name: str) -> Dict[str, Any]:
    """读取 mock JSON，失败返回空结构（不抛异常，保证工具链路稳定）"""
    path = _MOCK_DIR / file_name
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[Mock] 读取模拟数据失败：{path}，{e}")
        return {}


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def mock_check_device_status(device_id: Optional[str] = None,
                             printer_name: Optional[str] = None) -> Dict[str, Any]:
    """
    【Mock 数据源】查询打印机在线状态、队列长度与错误码。
    支持按 device_id 或 printer_name 查询；未找到时返回 hit=False，不抛异常。
    """
    key = _norm(device_id)
    name = _norm(printer_name)
    devices = _load_json("devices.json").get("devices", [])
    if not key and not name:
        available = "；".join(
            f"{d.get('device_id')}/{d.get('printer_name')}" for d in devices
        )
        return {
            "hit": False,
            "error": "缺少参数：device_id 或 printer_name 至少提供一个",
            "hint": f"可用设备：{available}",
        }

    for dev in devices:
        if key and _norm(dev.get("device_id")) == key:
            return _device_result(dev)
        if name and _norm(dev.get("printer_name")) == name:
            return _device_result(dev)

    return {
        "hit": False,
        "error": f"未找到设备：{device_id or printer_name}",
        "device_id": device_id or "",
        "printer_name": printer_name or "",
    }


def _device_result(dev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hit": True,
        "device_id": dev.get("device_id", ""),
        "printer_name": dev.get("printer_name", ""),
        "online": bool(dev.get("online")),
        "queue_length": int(dev.get("queue_length") or 0),
        "error_code": dev.get("error_code"),
        "last_seen": dev.get("last_seen", ""),
    }


def mock_check_user_permission(user_id: Optional[str], device_id: Optional[str]) -> Dict[str, Any]:
    """
    【Mock 数据源】查询用户对指定打印机的权限与驱动版本。
    未找到用户/设备/授权记录时返回 hit=False，不抛异常。
    """
    uid = _norm(user_id)
    did = _norm(device_id)
    if not uid or not did:
        return {"hit": False, "error": "缺少参数：user_id 与 device_id 均为必填"}

    records = _load_json("user_permissions.json").get("permissions", [])
    for rec in records:
        if _norm(rec.get("user_id")) == uid and _norm(rec.get("device_id")) == did:
            return {
                "hit": True,
                "user_id": rec.get("user_id", ""),
                "device_id": rec.get("device_id", ""),
                "has_permission": bool(rec.get("has_permission")),
                "driver_version": rec.get("driver_version", ""),
                "expected_version": rec.get("expected_version", ""),
            }

    return {
        "hit": False,
        "error": f"未找到用户 {user_id} 对设备 {device_id} 的权限记录",
        "user_id": user_id or "",
        "device_id": device_id or "",
    }
