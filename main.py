"""知喻工具箱 —— 自制多功能插件。

当前内置功能
1. 运行状态查询：管理员问「查状态/跑多久了/今天用了多少」等，直接发状态卡片，不进模型、不写会话历史、不进记忆库。
2. 发言清洗：发送前删占位词、括号动作、复述与拼贴，压波浪号、限长、去重，防刷屏防复读。
3. 发送节流：全局 60 秒窗口插话上限（口令「群聊上限N」）+ 单群随机间隔（口令「聊天间隔A-B」），被@不计数。
4. 群聊总闸：口令「关闭群聊/开启群聊」一键静默/恢复所有群。

设计要点
1. 不用斜杠命令：框架内置管理员命令已关闭，改由模型通过「工具调用」理解自然语言。
   模型负责辨认「对方是不是在问运行状态」，插件负责硬校验「对方是不是管理员」。
   两层必须分开，否则群里有人用话术就能骗模型泄露运行数据。
2. 这是「多功能工具箱」，状态查询只是里面第一个工具。以后加功能都加在本文件里，
   不新建插件、不重装：采集函数放「采集函数区」，本类里再加一个 @filter.llm_tool 方法，
   复用 self._check() 做权限校验即可。面板上的管理员名单和群聊开关是全工具箱通用的。
3. 状态查询的数据全部来自本机文件，不调用模型、不联网、不消耗字数；不输出密钥、地址、文件路径。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import difflib
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version as pkg_version

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

# ------------------------------------------------------------------ 常量区
# 数据目录自动定位：本插件装在 <data>/plugins/<本插件>/ 下，往上两级就是 data。
# 不写死 /AstrBot/data，Linux 容器、Windows、任意部署方式都能对上。
DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFO_FILE = os.path.join(DATA, "snowluma_info.json")
STATUS_FILE = os.path.join(DATA, "status.json")
DB_FILE = os.path.join(DATA, "data_v4.db")
CFG_FILE = os.path.join(DATA, "cmd_config.json")
LOG_FILE = os.path.join(DATA, "logs", "astrbot.log")
CST = timezone(timedelta(hours=8))
# 插件自有数据目录（市场审查要求：插件状态不得写进 data/config/）
PLUGIN_DATA_DIR = os.path.join(DATA, "plugin_data", "astrbot_plugin_zhiyu_toolbox")
GROUP_REPLY_OFF_FLAG = os.path.join(PLUGIN_DATA_DIR, ".zhiyu_group_reply_off")
GROUP_ALLOW_FILE = os.path.join(PLUGIN_DATA_DIR, ".zhiyu_group_allow")
FALLBACK_ADMINS = set()  # 兜底管理员：留空=完全跟随主配置

MODEL_LABELS = {
    "nvidia/nemotron-3.5-super-vl-120b-a12b": "看图大模型",
    "nvidia/nemotron-3.5-bolt-e4b": "轻量小模型",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "快速模型",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": "主力大模型",
    "deepseek/deepseek-flash": "深度求索快答",
    "deepseek-flash": "深度求索快答",
    "zhipu/glm-4v-flash": "免费看图档",
    "zhipu/glm-4-flash-250414": "免费文本档",
    "zhipu/glm-4-flash": "免费通用档",
}
GROUP_LABELS = {}  # 群号→显示名：留空=用框架里的群名

# -------------------------------------------------------------- 采集函数区
def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def _token(n) -> str:
    n = int(n or 0)
    return f"{n / 10000:.1f}万 token" if n >= 10000 else f"{n:,} token"


def _duration(seconds) -> str:
    """完整写成「2天1小时5分钟」这种，方便和旧状态卡片对照。"""
    seconds = max(0, int(seconds or 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    parts = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    parts.append(f"{rest // 60}分钟")
    return "".join(parts)


def _model_label(mid) -> str:
    if not mid:
        return "读不到"
    return MODEL_LABELS.get(str(mid), str(mid).rsplit("/", 1)[-1])


def _curr_selected_model_id() -> str:
    """用户在面板「模型提供商」里选择的当前使用模型（db preferences.curr_provider）。

    没选/读不到返回空串。这是运行时真正生效的对话模型来源，
    cmd_config 里 agent_runner.model 只是启动兜底，会被面板选择覆盖。
    """
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("select value from preferences where key='curr_provider'").fetchone()
        finally:
            con.close()
        val = row[0] if row else ""
        if isinstance(val, str) and val.strip():
            obj = json.loads(val) if val.strip().startswith("{") else {"val": val.strip()}
            return str(obj.get("val", "") or "")
        return ""
    except Exception:
        return ""


def _model_display() -> dict:
    """当前使用模型的统一显示结果：collect() 与 web_console_get() 共用这一份，
    全流程只读一次数据库，避免「两处各读一次、各叫一个名字」。

    返回字段：
      id        去首尾空格后的完整原始 ID（仅「未登记」态展示给用户），空则 ""
      label     给用户看的中文文案，绝不含英文尾串
      status    hit / unregistered / unselected / read_fail 四态
      read_fail 是否读取失败（数据库异常），与业务状态分开，不再被吞成空串
    """
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("select value from preferences where key='curr_provider'").fetchone()
        finally:
            con.close()
        val = row[0] if row else ""
        if isinstance(val, str) and val.strip():
            s = val.strip()
            if s.startswith("{"):
                obj = json.loads(s)
                raw = obj.get("val", "") if isinstance(obj, dict) else ""
            else:
                raw = s
        else:
            raw = ""
    except Exception:
        return {"id": "", "label": "暂未获取到模型", "status": "read_fail", "read_fail": True}

    rid = ("" if raw is None else str(raw)).strip()
    if (not rid) or rid == "关闭":
        return {"id": "", "label": "未选择模型", "status": "unselected", "read_fail": False}

    # 直接显示用户在「模型提供商」里选中的那个模型（完整模型名/ID），
    # 不再用「快速模型」这类中文别名代号代替，避免一进界面看不出选的是谁。
    return {"id": rid, "label": rid, "status": "hit", "read_fail": False}


def _effective_model(disp: dict, pid: str) -> dict:
    """状态栏显示的模型 = 第三关判断实际使用的模型。

    与陪伴插件 gate3 的解析口径一致（proactive_companion/main.py:1021-1044）：
      具体 ID -> 直接用；"@current" -> 跟随主对话 curr_provider；空 -> 不判定。
    """
    pid = str(pid or "").strip()
    if pid == "@current":
        return dict(disp or {})
    if pid:
        return {"id": pid, "label": pid, "status": "hit", "read_fail": False}
    return {"id": "", "label": "未选择模型", "status": "unselected", "read_fail": False}


def _framework_version() -> str:
    try:
        return pkg_version("astrbot")
    except Exception:
        return "读不到"


def _container_uptime() -> str:
    """用进程 1 的启动时刻算容器已运行多久，不依赖外部命令。"""
    try:
        with open("/proc/1/stat", encoding="utf-8") as f:
            parts = f.read().split()
        ticks = int(parts[21])
        hz = os.sysconf("SC_CLK_TCK") or 100
        with open("/proc/uptime", encoding="utf-8") as f:
            up = float(f.read().split()[0])
        return _duration(up - ticks / hz)
    except Exception:
        return "读不到"


def _uptime_from(raw) -> str:
    """把状态文件里的世界时时间戳换算成「几天几小时几分钟」。"""
    stamp = str(raw or "")[:19]
    if not stamp:
        return "读不到"
    try:
        dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return _duration((datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return "读不到"


def _qq_uptime(info: dict) -> str:
    """已上线时长：按 QQ 端容器创建时间算，也就是这个实例从哪天开始服役。

    中途短暂 stop/start 不会清零；容器被删除重建才会重新计时。
    """
    return _uptime_from(info.get("created_at") or info.get("started_at"))


def _qq_uptime_session(info: dict) -> str:
    """本次连续运行：按容器最近一次启动时间算，中途停过就重新开始。"""
    return _uptime_from(info.get("started_at"))


def _plugin_count() -> int:
    try:
        root = os.path.join(DATA, "plugins")
        return sum(1 for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    except Exception:
        return 0


def _status_service() -> str:
    """宿主状态服务把状态写进挂载目录里的 status.json，直接读文件最稳。

    若文件久未更新，只说「久未更新」，不把里面的旧状态当成当前状态报出来，
    免得机器明明在线却显示成离线。
    """
    try:
        d = _load_json(STATUS_FILE)
        if not d:
            return "读不到"
        ts = int(d.get("updated") or 0)
        if not ts:
            return "状态文件未更新"
        age = datetime.now().timestamp() - ts
        if age > 3600:
            return "状态文件久未更新（辅助服务未接入）"
        return f"{d.get('state') or '未知'}（{_duration(age)}前更新）"
    except Exception:
        return "读不到"


def _recent_errors():
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-300:]
        return sum(1 for line in tail if "[ERRO]" in line)
    except Exception:
        return None


def _today_stats() -> dict:
    now = datetime.now(CST)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    expr = "coalesce(token_input_other,0)+coalesce(token_input_cached,0)+coalesce(token_output,0)"
    empty = {
        "calls": 0, "total": 0, "kinds": {}, "by_model": [],
        "kind_calls": {}, "avg_kinds": {},
        "cached": 0, "input_total": 0, "hit_rate": 0.0,
        "rpm": 0.0, "now": now,
    }
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True, timeout=5)
        try:
            calls, total, other, cached = con.execute(
                f"select count(*), coalesce(sum({expr}),0), "
                f"coalesce(sum(token_input_other),0), coalesce(sum(token_input_cached),0) "
                f"from provider_stats where created_at>=?",
                (start_utc,),
            ).fetchone()
            kinds = dict(con.execute(
                f"select case when umo like '%GroupMessage%' then '群聊' "
                f"when umo like '%FriendMessage%' then '私聊' else '其它' end, "
                f"coalesce(sum({expr}),0) from provider_stats where created_at>=? group by 1",
                (start_utc,),
            ).fetchall())
            kind_calls = dict(con.execute(
                "select case when instr(umo,'GroupMessage')>0 then '群聊' "
                "when instr(umo,'FriendMessage')>0 then '私聊' else '其它' end, "
                "count(*) from provider_stats where created_at>=? group by 1",
                (start_utc,),
            ).fetchall())
            by_model = con.execute(
                f"select provider_model, coalesce(sum({expr}),0) from provider_stats "
                f"where created_at>=? group by 1 order by 2 desc limit 5",
                (start_utc,),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return empty
    minutes = max(1, int((now - start).total_seconds() // 60))
    input_total = int(other or 0) + int(cached or 0)
    hit_rate = (int(cached or 0) / input_total * 100) if input_total else 0.0
    return {
        "calls": int(calls or 0),
        "total": int(total or 0),
        "kinds": kinds,
        "kind_calls": kind_calls,
        "avg_kinds": {k: int(kinds.get(k, 0) / n) for k, n in kind_calls.items() if n},
        "by_model": by_model,
        "cached": int(cached or 0),
        "input_total": input_total,
        "hit_rate": hit_rate,
        "rpm": (calls or 0) / minutes,
        "now": now,
    }


def _astrbot_proc_uptime() -> str:
    """AstrBot 主进程运行时长（读 /proc），取不到时退回容器时长。"""
    try:
        import os as _os
        clk = _os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime", encoding="utf-8") as f:
            up = float(f.read().split()[0])
        best = None
        for pid in _os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode("utf-8", "replace")
            except Exception:
                continue
            if "astrbot" not in cmd.lower():
                continue
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                fields = f.read().split()
            st = int(fields[21]) / clk
            age = up - st
            if best is None or age > best:
                best = age
        if best is None:
            return _container_uptime()
        return _duration(best)
    except Exception:
        pass
    try:
        import psutil
        import datetime as _dt
        best = None
        for pr in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmd = " ".join(pr.info.get("cmdline") or [])
                name = str(pr.info.get("name") or "")
                if "astrbot" not in cmd.lower() and "astrbot" not in name.lower():
                    continue
                age = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromtimestamp(pr.info["create_time"], _dt.timezone.utc)).total_seconds()
                if best is None or age > best:
                    best = age
            except Exception:
                continue
        if best is None:
            return "读不到"
        return _duration(best)
    except Exception:
        return _container_uptime()


# [2026-11-15] WebUI 控制台视图：允许在线修改的配置键与类型钳制
CONSOLE_CONFIG_KEYS = (
    "admin_ids", "allow_group", "delivery",
    "clean_enabled", "clean_scope", "clean_block_phrases", "clean_bracket_keywords",
    "clean_strip_all_brackets", "clean_strip_unclosed_brackets", "clean_strip_markup_tags",
    "clean_dedupe_seconds", "clean_max_lines", "clean_wave_max", "clean_min_chars",
    "clean_max_chars", "clean_custom_rules", "clean_extra_strip_words",
    "clean_skip_prefixes", "clean_verbose_log",
    "send_rate_enabled", "send_rate_window", "send_rate_max",
    "group_gap_enabled", "group_gap_min_seconds", "group_gap_max_seconds",
)
_CONSOLE_INT_RANGE = {
    "clean_dedupe_seconds": (0, 86400),
    "clean_max_lines": (0, 20),
    "clean_wave_max": (0, 10),
    "clean_min_chars": (0, 100),
    "clean_max_chars": (0, 2000),
    "send_rate_window": (1, 3600),
    "send_rate_max": (1, 999),
    "group_gap_min_seconds": (0, 3600),
    "group_gap_max_seconds": (0, 3600),
}
_CONSOLE_BOOL_KEYS = {
    "allow_group", "clean_enabled", "clean_strip_all_brackets",
    "clean_strip_unclosed_brackets", "clean_strip_markup_tags", "clean_verbose_log",
    "send_rate_enabled", "group_gap_enabled",
}
_CONSOLE_ENUM = {
    "delivery": ("card", "model"),
    "clean_scope": ("all", "group", "private"),
}

# 新开关在旧配置里不存在时按「开」处理，保持升级前的原有行为。
_CONSOLE_BOOL_DEFAULT = {
    "send_rate_enabled": True,
    "group_gap_enabled": True,
}


def _console_sanitize(key, val):
    """按 schema 类型钳制控制台提交来的配置值；不合法返回 None（不覆盖）。"""
    try:
        if key in _CONSOLE_BOOL_KEYS:
            return bool(val)
        if key in _CONSOLE_INT_RANGE:
            lo, hi = _CONSOLE_INT_RANGE[key]
            return max(lo, min(hi, int(val)))
        s = str(val)
        if key in _CONSOLE_ENUM:
            return s if s in _CONSOLE_ENUM[key] else None
        return s
    except Exception:
        return None


def collect() -> dict:
    """汇总所有状态数据。以后加新功能时，可复用这里的采集结果。"""
    info = _load_json(INFO_FILE)
    cfg = _load_json(CFG_FILE)
    runner = ((cfg.get("agent_runner") or {}).get("config") or {})
    disp = _model_display()
    return {
        "has_sl": bool(info),
        "framework": _framework_version(),
        "qq_version": info.get("version") or "读不到",
        "qq_platform": info.get("platform") or "读不到",
        "qq_uptime": _qq_uptime(info),
        "system_uptime": _uptime_from(info.get("system_booted_at")),
        "qq_uptime_session": _qq_uptime_session(info),
        "container_uptime": _container_uptime(),
        "astrbot_uptime": _astrbot_proc_uptime(),
        "plugins": _plugin_count(),
        "status_service": _status_service(),
        "default_model": (disp["id"] or "关闭"),
        "model_display": disp,
        "limits": info.get("rate_limits_today") if isinstance(info.get("rate_limits_today"), int) else 0,
        "errors": _recent_errors(),
        "stats": _today_stats(),
    }


# [2026-11-11] 管理员查状态直出：命中问法且是管理员时直接发卡片并终止事件，
# 不进模型、不写会话历史、不进记忆库。关键词从严，没命中仍走原 llm_tool 兜底。
STATUS_DIRECT_RE = re.compile(
    r"(查下状态|查一下状态|查状态|看下状态|看看状态|查看状态|状态卡|运行状态|"
    r"跑[了得]?多久|跑得怎么样|用了多少[ ]*token|用了多少字|"
    r"装了几个插件|服务还正常|有没有报错|限流次数|哪个模型最费|"
    r"今天用了多少|现在用的是哪个模型)"
)

def format_card(d: dict) -> str:
    # 2026-11-12 用户样板 v2：带分隔线，新增群/私聊次数与 LLM 调用次数。
    # 均句消耗 = 该会话类型今日总 token 除以该类型请求次数，纯数量平均，不按缓存价格折算。
    # 数字列对齐：带「万」的宽值用单制表符，窄值用双制表符，token 列自动对齐。
    s = d["stats"]
    kind_calls = s.get("kind_calls") or {}
    avg = s.get("avg_kinds") or {}

    def _t(n) -> str:
        n = int(n or 0)
        return f"{n / 10000:.2f}万" if n >= 10000 else f"{n:,}"

    def _avg(k):
        return _t(avg.get(k) or 0)

    def _row(label, val):
        sep = "\t" if "万" in val else "\t\t"
        return f"{label}：{val}{sep}tokens"

    hh = f"{s['now'].hour:02d}"
    mm = f"{s['now'].minute:02d}"
    pct = chr(37)
    EQ = "=" * 23
    DASH = "-" * 40
    head = ["SL 状态" if d.get("has_sl") else "运行状态", EQ]
    if d.get("has_sl"):
        head += [f"版本: SL {d['qq_version']}", f"平台: {d['qq_platform']}"]
    head.append(f"系统运行: {d['system_uptime']}")
    if d.get("has_sl"):
        head.append(f"SL 进程: {d['qq_uptime_session']}")
    head += [
        f"AS状态: {d.get('astrbot_uptime') or d['container_uptime']}",
        EQ,
    ]
    return "\n".join(head + [
        "今日token用量：",
        f"（直至 {hh}:{mm}）",
        f"群聊次数：{int(kind_calls.get('群聊', 0))}次",
        f"私聊次数：{int(kind_calls.get('私聊', 0))}次",
        DASH,
        _row("群聊一共消耗", _t(s['kinds'].get('群聊', 0))),
        _row("群聊均句消耗", _avg('群聊')),
        _row("私聊一共消耗", _t(s['kinds'].get('私聊', 0))),
        _row("私聊均句消耗", _avg('私聊')),
        DASH,
        f"总用量： {_t(s['total'])}\ttokens",
        f"调用：{int(kind_calls.get('群聊', 0)) + int(kind_calls.get('私聊', 0))}次",
        f"LLM：{int(s['calls'])}次",
        f"缓存命中率： {s['hit_rate']:.1f}" + pct,
        f"RPM： {s['rpm']:.1f}",
        f"总用量： {_t(s['total'])}\ttokens",
        EQ,
    ])



def render_status_image(d: dict):
    """用 Pillow 把状态卡画成田字格表格图片，返回 PNG 字节；PIL 不可用时返回 None。

    纯本地绘图：数据来自 collect()，不调用模型、不联网、不耗 token。
    字体用容器自带的 Noto Sans CJK；对齐为像素级，不受 QQ 字体影响。
    """
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    s = d["stats"]
    kind_calls = s.get("kind_calls") or {}
    avg = s.get("avg_kinds") or {}

    def _t(n) -> str:
        n = int(n or 0)
        return f"{n / 10000:.2f}万" if n >= 10000 else f"{n:,}"

    rows_a = []
    if d.get("has_sl"):
        rows_a.append(("版本", f"SL {d['qq_version']}"))
        rows_a.append(("平台", str(d["qq_platform"])))
    rows_a.append(("系统运行", str(d["system_uptime"])))
    if d.get("has_sl"):
        rows_a.append(("SL 进程", str(d["qq_uptime_session"])))
    rows_a.append(("AS状态", str(d.get("astrbot_uptime") or d["container_uptime"])))
    rows_a.append(("插件", str(d["plugins"])))
    rows_b = [
        ("群聊", f"{int(kind_calls.get('群聊') or 0)} 次 · {_t(s['kinds'].get('群聊', 0))} tokens"),
        ("私聊", f"{int(kind_calls.get('私聊') or 0)} 次 · {_t(s['kinds'].get('私聊', 0))} tokens"),
        ("均句", f"群 {_t(avg.get('群聊', 0))} · 私 {_t(avg.get('私聊', 0))}"),
        ("调用", f"{int(kind_calls.get('群聊') or 0) + int(kind_calls.get('私聊') or 0)} 次"),
        ("LLM", f"{int(s['calls'])} 次"),
        ("缓存", f"{s['hit_rate']:.1f}%"),
        ("RPM", f"{s['rpm']:.1f}"),
        ("总耗", f"{_t(s['total'])} tokens"),
    ]
    hh = f"{s['now'].hour:02d}"
    mm = f"{s['now'].minute:02d}"

    W, M = 680, 20
    col1 = 190
    row_h, head_h, title_h, foot_h = 54, 58, 84, 46
    rows = rows_a + rows_b
    H = title_h + head_h + row_h * len(rows) + foot_h
    img = Image.new("RGB", (W, H), "#0b0e13")
    dr = ImageDraw.Draw(img)
    def _font(size):
        bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        if os.path.isdir(bundled):
            for n in sorted(os.listdir(bundled)):
                if n.lower().endswith((".ttf", ".otf", ".ttc")):
                    try:
                        return ImageFont.truetype(os.path.join(bundled, n), size)
                    except Exception:
                        continue
        names = [
            "NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf",
            "NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf",
            "msyhbd.ttc", "msyh.ttc", "simhei.ttf", "simsun.ttc",
        ]
        dirs = [
            "/usr/share/fonts/opentype/noto/",
            "/usr/share/fonts/truetype/noto/",
            "/usr/share/fonts/noto-cjk/",
            "C:/Windows/Fonts/",
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/"),
        ]
        for d in dirs:
            for n in names:
                fp = d + n
                if os.path.exists(fp):
                    try:
                        return ImageFont.truetype(fp, size)
                    except Exception:
                        continue
        return None
    f_title = _font(38)
    f_head = _font(27)
    f_cell = _font(26)
    f_small = _font(22)
    if not (f_title and f_head and f_cell and f_small):
        return None

    dr.text((W // 2, 18), "机器人运行状态", font=f_title, fill="#f8fafc", anchor="ma")

    ty = title_h - 12
    tw = W - M * 2
    bh = head_h + row_h * len(rows)
    dr.rectangle([M, ty, M + tw, ty + bh], fill="#151a21")
    dr.rectangle([M, ty, M + tw, ty + head_h], fill="#232a33")

    line = "#566171"
    dr.line([M, ty, M + tw, ty], fill=line, width=2)
    dr.line([M, ty + bh, M + tw, ty + bh], fill=line, width=2)
    dr.line([M, ty + head_h, M + tw, ty + head_h], fill=line, width=2)
    split_y = ty + head_h + row_h * len(rows_a)
    dr.line([M, split_y, M + tw, split_y], fill=line, width=3)
    vx = M + col1
    dr.line([vx, ty, vx, ty + bh], fill=line, width=2)

    def cell(x, cy, text, font, color="#e5e7eb"):
        dr.text((x, cy), text, font=font, fill=color, anchor="lm")

    cy = ty + head_h // 2
    cell(M + 16, cy, "项目", f_head, "#9aa4b2")
    cell(vx + 16, cy, "当前", f_head, "#9aa4b2")
    for i, (k, v) in enumerate(rows):
        ry = ty + head_h + row_h * i
        if i:
            dr.line([M, ry, M + tw, ry], fill="#2c3440", width=1)
        cy = ry + row_h // 2
        cell(M + 16, cy, k, f_cell, "#9aa4b2")
        cell(vx + 16, cy, v, f_cell, "#f8fafc")

    dr.text((W // 2, ty + bh + (H - ty - bh) // 2), f"截至 {hh}:{mm}", font=f_small, fill="#9aa4b2", anchor="ma")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()



# ---------------------------------------------- 第三个功能：发言清洗（发送前）
# 全部规则都能在面板「知喻工具箱」设置里改，改完立即生效，不用重启。
_CLEAN_BLOCK_DEFAULT = (
    "（安静旁听，不回复）,（安静旁听）,（不回）,（不回复）,安静旁听，不回复,"
    "我选择不回复,（沉默）,保持沉默"
)
_CLEAN_BRACKET_DEFAULT = (
    "探头,旁听,不回,安静,沉默,笑,摊手,挠头,歪头,眨眼,小声,轻声,顿了顿,愣,叹气,"
    "脸红,害羞,凑近,转身,抱膝,趴,嘟囔,嘀咕,扶额,摇头,点头"
)
_CLEAN_SKIP_DEFAULT = ""
_CLEAN_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_CLEAN_BRACKET_RE = re.compile(r"[（(【\[][^）)】\]]{0,40}[）)】\]]")
_CLEAN_STAR_RE = re.compile(r"\*[^*\n]{1,40}\*")
_CLEAN_PUNCT_RE = re.compile(r"[\s，,。.!！?？~～、；;…·\-—]+")
_CLEAN_REFUSE_RE = re.compile(r"</?\s*refuse\s*/?>", re.I)
# 模型偶尔自造的包裹标签（如 <reply>…</reply>）：开/闭、成对/半截都删，
# 防止字面标签发进 QQ、再写进会话历史被模型照抄扩散。
_CLEAN_FAKE_TAG_RE = re.compile(r"</?\s*(?:reply|think|thought|output|response|answer|speech|say|msg|message)\s*/?>", re.I)
_CLEAN_DEDUPE_JUNK_RE = re.compile(r"[（）()\[\]【】\"'“”‘’:：#*&^/\\|]+")


def _dedupe_key(text: str) -> str:
    """把一句话压成比较用的骨架：去掉表情、标点和波浪号，只留实质内容。

    这样「好呀~」「好呀！😊」「好 呀」会算同一句，但意思不同的话不会误判。
    """
    s = _CLEAN_EMOJI_RE.sub("", str(text or ""))
    s = s.replace("\ufe0f", "").replace("\u200d", "")
    s = _CLEAN_DEDUPE_JUNK_RE.sub("", s)
    return _CLEAN_PUNCT_RE.sub("", s).lower()


def _clean_words(raw, default: str) -> list:
    """把设置框里用逗号或换行分隔的词，拆成干净的词表。"""
    return [x.strip() for x in re.split(r"[,\n]+", str(raw if raw is not None else default)) if x.strip()]


def _clean_int(cfg, key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(cfg.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def clean_outgoing_text(text: str, cfg: dict) -> tuple:
    """纯函数：清洗一条发言。返回 (清洗后文本, 动作列表)；空文本代表整条不发送。

    只处理文字，不碰图片等其它消息组件；不调用模型、不联网、不写文件。
    """
    actions = []
    s = str(text or "").strip()
    if not s:
        return "", actions
    for prefix in _clean_words(cfg.get("clean_skip_prefixes"), _CLEAN_SKIP_DEFAULT):
        if s.startswith(prefix):
            return s, actions  # 状态卡片等结构化内容原样放行

    # 0.5) <refuse/> 标签：模型自主选择这轮不说话；去掉标签后无实质内容就不发送
    refuse_rest = _CLEAN_REFUSE_RE.sub("", s)
    if refuse_rest != s and not _CLEAN_PUNCT_RE.sub("", refuse_rest):
        return "", ["<refuse/> 主动拒答"]

    # 0.6) 删模型自造的包裹标签（见 _CLEAN_FAKE_TAG_RE 注释）
    s2 = _CLEAN_FAKE_TAG_RE.sub("", s)
    if s2 != s:
        s = s2.strip()
        actions.append("删自造标签")

    # 1) 占位词：整条只是「不回复」的意思时，直接不发
    blocks = _clean_words(cfg.get("clean_block_phrases"), _CLEAN_BLOCK_DEFAULT)
    probe = s
    for word in blocks:
        probe = probe.replace(word, "")
    if not _CLEAN_PUNCT_RE.sub("", probe):
        return "", ["占位不发送"]
    for word in blocks:
        if word in s:
            s = s.replace(word, "")
            actions.append("去占位")

    # 2) 括号 / 星号包着的动作神态
    strip_all = bool(cfg.get("clean_strip_all_brackets", False))
    keywords = _clean_words(cfg.get("clean_bracket_keywords"), _CLEAN_BRACKET_DEFAULT)

    def _bracket_sub(match):
        chunk = match.group(0)
        return "" if (strip_all or any(k in chunk for k in keywords)) else chunk

    s2 = _CLEAN_BRACKET_RE.sub(_bracket_sub, s)
    if s2 != s:
        s = s2
        actions.append("删括号动作")
    s2 = _CLEAN_STAR_RE.sub("", s)
    if s2 != s:
        s = s2
        actions.append("删星号动作")
    # 没写完的半截括号动作（如「(笑死」），只在整条末尾时才删，避免误伤正文
    if bool(cfg.get("clean_strip_unclosed_brackets", True)):
        s2 = re.sub(r"[（(【\[][^）)】\]]{0,20}$", "", s)
        if s2 != s:
            s = s2
            actions.append("删半截括号")
    # 表情包插件的分类标签（完整 &&happy&& 或残缺 &&see），用本地正则清，
    # 不交给陪伴插件的正则执行器（那边有执行预算，长文本会超时导致发送失败）。
    if bool(cfg.get("clean_strip_markup_tags", True)):
        s2 = re.sub(r"&&[A-Za-z0-9_-]*&{0,2}", "", s)
        if s2 != s:
            s = s2
            actions.append("删配图标签")

    # 3) 自己写的替换规则：每行「查找=>替换」，替换留空就是删掉
    for line in re.split(r"[;\n]+", str(cfg.get("clean_custom_rules") or "")):
        if "=>" not in line:
            continue
        find, replace = (part.strip() for part in line.split("=>", 1))
        if find and find in s:
            s = s.replace(find, replace)
            actions.append(f"自定义:{find}")

    # 3.5) 用户自定义清洗词/标签：命中直接删（普通词或 <标签> 都行）
    extra_words = _clean_words(cfg.get("clean_extra_strip_words"), "")
    if extra_words:
        s2 = s
        for w in extra_words:
            s2 = s2.replace(w, "")
        if s2 != s:
            s = s2.strip()
            actions.append("删自定义词")

    # 4) 空行合并 + 最多几行
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s2 = re.sub(r"\n{2,}", "\n", s)
    if s2 != s:
        s = s2
        actions.append("并空行")
    lines = [part.strip() for part in s.split("\n") if part.strip()]
    max_lines = _clean_int(cfg, "clean_max_lines", 2, 0, 10)
    if max_lines and len(lines) > max_lines:
        if max_lines == 1:
            lines = [" ".join(lines)]
        else:
            lines = [" ".join(lines[: max_lines - 1]), " ".join(lines[max_lines - 1 :])]
        actions.append("并段")
    s = "\n".join(lines).strip()

    # 5) 波浪号只留前 N 个
    wave_max = _clean_int(cfg, "clean_wave_max", 1, 0, 10)
    kept, buf = 0, []
    for ch in s:
        if ch in "~～":
            kept += 1
            if kept <= wave_max:
                buf.append(ch)
            elif "压波浪号" not in actions:
                actions.append("压波浪号")
        else:
            buf.append(ch)
    s = re.sub(r"[ \t]{2,}", " ", "".join(buf)).strip()

    # 5.5) 人设要求句末不写句号：只剥结尾的一个句号，不动句中标点
    s = s.rstrip()
    if s.endswith("。"):
        s = s[:-1].rstrip()
        actions.append("剥句尾句号")


    # 6) 一条最多多少字：超了优先保留完整的第一句，再不行才硬截
    max_chars = _clean_int(cfg, "clean_max_chars", 50, 0, 500)
    if max_chars and len(s) > max_chars:
        cut = s[:max_chars]
        stop = re.search(r"[。！？!?]", cut)
        if stop and stop.end() >= 10:
            s = cut[: stop.end()]
        else:
            # 人设卡要求句末不写句号，句号断点经常不存在；退一步在最后一个
            # 逗号/顿号/分号处断开，保住完整分句，避免把话切在半截。
            breaks = [cut.rfind(ch) for ch in "，,、；;"]
            comma = max(breaks) if breaks else -1
            if comma >= 15:
                s = cut[:comma]
            else:
                s = cut.rstrip("，,、；; ")
        actions.append(f"限长{max_chars}字")

    # 7) 太短 / 清洗后没内容，就不发
    min_chars = _clean_int(cfg, "clean_min_chars", 0, 0, 50)
    plain = _CLEAN_PUNCT_RE.sub("", s)
    if not plain:
        return "", actions + ["清洗后为空不发送"]
    if min_chars and len(plain) < min_chars:
        return "", actions + [f"短于{min_chars}字不发送"]
    return s, actions


# ---------------------------------------------------------------- 插件主体
@register(
    "astrbot_plugin_zhiyu_toolbox",
    "大肥鱼",
    "知喻的多功能工具箱：状态查询、发言清洗、发送节流、群聊总闸，支持聊天口令直改，带 WebUI 控制台视图",
    "1.11.1",
)
class ZhiyuToolbox(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        # 合规迁移：旧版把总闸/白名单标记写在 data/config/，按审查要求挪到 plugin_data/
        try:
            os.makedirs(PLUGIN_DATA_DIR, exist_ok=True)
            for _old, _new in (
                (os.path.join(DATA, "config", ".zhiyu_group_reply_off"), GROUP_REPLY_OFF_FLAG),
                (os.path.join(DATA, "config", ".zhiyu_group_allow"), GROUP_ALLOW_FILE),
            ):
                if os.path.exists(_old) and not os.path.exists(_new):
                    os.replace(_old, _new)
        except Exception as exc:
            logger.warning("[zhiyu_toolbox] 标记文件迁移失败: " + repr(exc))
        # 复述检测用的短期记忆：按会话记最近收到的原话、自己发出去的原话
        self._recent_src_text = {}
        self._recent_own_text = {}
        self._clean_count = 0
        # send_message_to_user 走框架旁路直发，普通回复的节流闸管不到它，这里补上
        self._patch_smtu_throttle()
        # WebUI 控制台视图（views/console/index.html）的后端接口
        self._register_console_api()

    def _patch_smtu_throttle(self):
        """接管框架工具 send_message_to_user：它不经过 on_decorating_result，
        普通回复的频率/间隔闸管不到；这里在工具入口前做同样的节流判定。"""
        try:
            from astrbot.core.tools.message_tools import SendMessageToUserTool
        except Exception as exc:
            logger.warning(f"[zhiyu_toolbox] 未能导入 send_message_to_user，节流未接管: {exc!r}")
            return
        if getattr(SendMessageToUserTool, "_zhiyu_throttled", False):
            return
        original_call = SendMessageToUserTool.call

        async def throttled_call(tool_self, context, **kwargs):
            ok, reason = self._smtu_throttle_check(context, kwargs)
            if not ok:
                return f"error: {reason}"
            return await original_call(tool_self, context, **kwargs)

        SendMessageToUserTool.call = throttled_call
        SendMessageToUserTool._zhiyu_throttled = True
        logger.info("[zhiyu_toolbox] 已接管 send_message_to_user：群聊插话同样受频率/间隔限制")

    def _smtu_throttle_check(self, context, kwargs) -> tuple[bool, str]:
        """send_message_to_user 的节流判定，与普通回复共用同一套窗口/间隔账本。
        返回 (是否放行, 拦截原因)。"""
        try:
            event = context.context.event
            umo = str(event.unified_msg_origin or "")
        except Exception:
            return True, ""
        if "GroupMessage" not in umo:
            return True, ""
        try:
            self_id = str(event.get_self_id() or "")
            msgs = getattr(getattr(event, "message_obj", None), "message", None) or []
            for mc in msgs:
                if type(mc).__name__ == "At" and str(getattr(mc, "qq", "")) == self_id:
                    return True, ""
        except Exception:
            pass
        gid = umo.rsplit(":", 1)[-1]
        session = str(kwargs.get("session") or "")
        if session and ":" in session:
            parts = session.split(":")
            if len(parts) >= 3 and "GroupMessage" in parts[1]:
                gid = parts[-1]
        win = _clean_int(self.config, "send_rate_window", 60, 1, 3600)
        cap = _clean_int(self.config, "send_rate_max", 4, 1, 999)
        gmin = _clean_int(self.config, "group_gap_min_seconds", 0, 0, 3600)
        gmax = _clean_int(self.config, "group_gap_max_seconds", 0, 0, 3600)
        if gmax < gmin:
            gmax = gmin
        now = time.monotonic()
        gap = random.randint(gmin, gmax) if (gmin and gid) else 0
        lastg = getattr(self, "_last_gap_send", None)
        if lastg is None:
            lastg = {}
            self._last_gap_send = lastg
        if gap and gid and now - lastg.get(gid, 0.0) < gap:
            logger.info(f"[zhiyu_toolbox] send_message_to_user 拦截：单群间隔未到（{gid}）")
            return False, "群聊插话间隔未到，本条未发送，请稍后再试"
        if gap and gid:
            lastg[gid] = now
        q = getattr(self, "_send_times", None)
        if q is None:
            q = []
            self._send_times = q
        while q and now - q[0] > win:
            q.pop(0)
        if len(q) >= cap:
            logger.info(f"[zhiyu_toolbox] send_message_to_user 拦截：插话超速（{win}s内{cap}条）")
            return False, "群聊发言频率已达上限，本条未发送，请稍后再试"
        q.append(now)
        return True, ""

    def _admins(self) -> set[str]:
        """优先用插件配置，留空则跟随主配置的管理员名单。"""
        raw = str(self.config.get("admin_ids") or "").strip()
        if raw:
            return {x.strip() for x in raw.split(",") if x.strip()}
        ids = _load_json(CFG_FILE).get("admins_id") or []
        return {str(x) for x in ids} or set(FALLBACK_ADMINS)

    def _check(self, event: AstrMessageEvent, scope: str = "admin", allow_in_group=None):
        """工具箱通用权限硬校验，以后每个新功能开头都调它，绝不交给模型判断。

        scope：
          admin    —— 默认。只认管理员，身份用真实 QQ 号比对，模型说了不算。
          everyone —— 所有人都能用，只给不带任何运行数据的功能（如闲聊小工具）。
        allow_in_group：
          None     —— 默认。跟随面板上「允许在群里使用管理工具」这个总开关。
          True/False —— 单个功能要强制指定时才传，一般不用。
        """
        if scope != "admin":
            return True, ""
        sender = str(event.get_sender_id() or "")
        if sender not in self._admins():
            return False, "对方不是管理员"
        group = str(event.get_group_id() or "")
        allow = bool(self.config.get("allow_group", False)) if allow_in_group is None else bool(allow_in_group)
        if group and not allow:
            return False, "当前设置只允许私聊使用"
        return True, ""

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=400000)
    async def group_reply_gate(self, event: AstrMessageEvent):
        # 群聊回复总闸（两级）：
        # 1) .zhiyu_group_reply_off 存在 → 所有群静默；
        # 2) .zhiyu_group_allow 存在 → 只放行文件里列出的群号，其余群消息直接终止。
        # priority 400000 高于陪伴插件 240000 与状态卡 300000。
        try:
            if os.path.exists(GROUP_REPLY_OFF_FLAG):
                event.stop_event()
                return
            if os.path.exists(GROUP_ALLOW_FILE):
                try:
                    with open(GROUP_ALLOW_FILE, encoding="utf-8") as fh:
                        allowed = {ln.strip() for ln in fh if ln.strip()}
                except Exception:
                    allowed = None
                if allowed:
                    gid = str(event.get_group_id() or "")
                    if gid and gid not in allowed:
                        event.stop_event()
        except Exception:
            pass


    @filter.event_message_type(filter.EventMessageType.ALL, priority=450000)
    async def group_gate_switch(self, event: AstrMessageEvent):
        # 管理员口令入口(450000 先于总闸 400000 执行):
        # 1) 关闭群聊/开启群聊 → 增删总闸 flag 文件(总闸读闸的写入口)。
        # 2) 聊天间隔A-B / 聊天间隔A → 设置单群随机插话间隔(秒),立即生效并落盘。
        # 3) 关闭聊天间隔 → 单群间隔置 0;查看聊天间隔 → 报当前区间。
        try:
            raw = str(event.get_message_str() or "")
            text = re.sub(r"\[.*?\]", "", raw).strip()
            m = re.match(r"^聊天间隔\s*(\d+)(?:\s*-\s*(\d+))?$", text)
            if text in ("关闭聊天间隔", "聊天间隔关闭"):
                ok, _ = self._check(event)
                if not ok:
                    return
                self.config["group_gap_min_seconds"] = 0
                self.config["group_gap_max_seconds"] = 0
                self._persist_config()
                await event.send(event.plain_result("单群插话间隔已关闭,恢复固定节奏~"))
                event.stop_event()
                logger.info("[zhiyu_toolbox] 聊天间隔调整: 关闭")
                return
            if text in ("查看聊天间隔", "聊天间隔多少"):
                ok, _ = self._check(event)
                if not ok:
                    return
                lo = _clean_int(self.config, "group_gap_min_seconds", 0, 0, 3600)
                hi = _clean_int(self.config, "group_gap_max_seconds", 0, 0, 3600)
                if lo <= 0:
                    msg = "当前没有单群间隔限制(已关闭)~"
                else:
                    msg = "当前单群插话间隔:%d~%d 秒(随机)~" % (lo, hi)
                await event.send(event.plain_result(msg))
                event.stop_event()
                return
            if m:
                ok, _ = self._check(event)
                if not ok:
                    return
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) else lo
                if hi < lo:
                    lo, hi = hi, lo
                if lo <= 0:
                    self.config["group_gap_min_seconds"] = 0
                    self.config["group_gap_max_seconds"] = 0
                    msg = "单群插话间隔已关闭,恢复固定节奏~"
                else:
                    self.config["group_gap_min_seconds"] = lo
                    self.config["group_gap_max_seconds"] = hi
                    msg = "单群插话间隔已设为 %d~%d 秒(随机)~" % (lo, hi)
                self._persist_config()
                await event.send(event.plain_result(msg))
                event.stop_event()
                logger.info("[zhiyu_toolbox] 聊天间隔调整: " + msg)
                return
            # 群聊每分钟硬上限口令：群聊上限N 设置;查看群聊上限/群聊上限多少 报当前值
            mcap = re.match(r"^群聊上限\s*(\d+)$", text)
            if mcap:
                ok, _ = self._check(event)
                if not ok:
                    return
                cap = max(1, min(999, int(mcap.group(1))))
                self.config["send_rate_max"] = cap
                self._persist_config()
                await event.send(event.plain_result("群聊每分钟插话上限已设为 %d 条~" % cap))
                event.stop_event()
                logger.info("[zhiyu_toolbox] 群聊上限调整: %d 条/分钟" % cap)
                return
            if text in ("查看群聊上限", "群聊上限多少"):
                ok, _ = self._check(event)
                if not ok:
                    return
                cap = _clean_int(self.config, "send_rate_max", 4, 1, 999)
                win = _clean_int(self.config, "send_rate_window", 60, 1, 3600)
                await event.send(event.plain_result(
                    "当前群聊插话上限:%d 条/%d 秒(所有群共享,被@不计数)~" % (cap, win)))
                event.stop_event()
                return
            if text not in ("关闭群聊", "开启群聊", "打开群聊"):
                return
            ok, _ = self._check(event)
            if not ok:
                return
            if text == "关闭群聊":
                with open(GROUP_REPLY_OFF_FLAG, "w", encoding="utf-8") as fh:
                    fh.write("off")
                await event.send(event.plain_result("群聊总闸已关闭,所有群静默~"))
            else:
                if os.path.exists(GROUP_REPLY_OFF_FLAG):
                    os.remove(GROUP_REPLY_OFF_FLAG)
                await event.send(event.plain_result("群聊总闸已开启,恢复群聊~"))
            event.stop_event()
            logger.info("[zhiyu_toolbox] 群聊总闸开关: " + text)
        except Exception as exc:
            logger.warning("[zhiyu_toolbox] 总闸开关失败: " + repr(exc))

    def _persist_config(self):
        # 把 self.config 当前值写回配置文件,改完重启也不丢。
        # 统一走框架 save_config() 正规途径,不再手写 data/config 文件(市场审查要求)。
        try:
            if hasattr(self.config, "save_config"):
                self.config.save_config()
            else:
                logger.warning("[zhiyu_toolbox] 当前框架 config 不支持 save_config(),配置未落盘")
        except Exception as exc:
            logger.warning("[zhiyu_toolbox] 配置落盘失败: " + repr(exc))


    # ------------------------------------------------- WebUI 控制台视图
    def _register_console_api(self):
        # 视图页通过 bridge 调 /api/v1/plugins/extensions/<插件名>/console；
        # 同时注册全名与短名路由，兼容不同入口的 pluginName 传参。
        routes = {
            "/astrbot_plugin_zhiyu_toolbox/console",
            "/zhiyu_toolbox/console",
        }
        try:
            for route in routes:
                self.context.register_web_api(
                    route, self.web_console_get, ["GET"], "工具箱控制台：读状态与配置")
                self.context.register_web_api(
                    route, self.web_console_post, ["POST"], "工具箱控制台：保存配置/总闸开关")
            logger.info("[zhiyu_toolbox] 控制台视图 API 已注册: " + ", ".join(sorted(routes)))
        except Exception as exc:
            logger.warning("[zhiyu_toolbox] 控制台视图 API 注册失败: " + repr(exc))

    def _companion_star(self):
        """拿主动陪伴插件实例，用于跨插件读写第三关裁判模型。"""
        try:
            md = self.context.get_registered_star("astrbot_plugin_proactive_companion")
            return getattr(md, "star_cls", None)
        except Exception:
            return None

    def _chat_model_options(self):
        """全部对话模型实例（provider/model），供裁判模型下拉选择。"""
        out = []
        try:
            pm = getattr(self.context, "provider_manager", None)
            for inst in list(getattr(pm, "provider_insts", None) or []):
                try:
                    m = inst.meta()
                    mid = str(getattr(m, "id", "") or "")
                    if mid:
                        out.append({"id": mid, "label": mid})
                except Exception:
                    continue
        except Exception:
            pass
        out.sort(key=lambda x: x["id"])
        return out

    async def web_console_get(self):
        try:
            snap = collect()
            st = snap.get("stats")
            if isinstance(st, dict) and st.get("now") is not None:
                st["now"] = str(st["now"])
            cfg = {k: self.config.get(k, _CONSOLE_BOOL_DEFAULT.get(k)) for k in CONSOLE_CONFIG_KEYS}
            comp = self._companion_star()
            ccfg = getattr(comp, "config", None) or {}
            gate3 = {"found": comp is not None, "enabled": bool(ccfg.get("gate3_enabled", True)),
                     "provider_id": str(ccfg.get("gate3_provider_id", "") or "")}
            # 概率海选：主动回复开关 + 可自填概率，均来自主动陪伴插件配置。
            try:
                _prob = float(ccfg.get("proactive_reply_probability", 0.35) or 0)
            except (TypeError, ValueError):
                _prob = 0.35
            proactive = {
                "enabled": bool(ccfg.get("proactive_reply_enabled", True)),
                "probability": max(0.0, min(1.0, _prob)),
            }
            disp = snap.get("model_display") or {}
            gate3["current_model_id"] = disp.get("id", "")
            gate3["current_model_label"] = disp.get("label") or "暂未获取到模型"

            # 状态栏「当前使用模型」与第三关判断模型同源，规则见 _effective_model()。
            _eff = _effective_model(disp, gate3.get("provider_id", ""))
            snap["model_display"] = _eff
            gate3["effective_model_id"] = _eff.get("id", "")
            gate3["effective_model_label"] = _eff.get("label", "")

            if gate3["provider_id"] == "@current":
                gate3["provider_display"] = "当前使用模型（" + gate3["current_model_label"] + "）"
            elif gate3["provider_id"]:
                gate3["provider_display"] = gate3["provider_id"]
            else:
                gate3["provider_display"] = ""
            return {"status": "ok", "data": {
                "version": "1.11.1",
                "gate_off": os.path.exists(GROUP_REPLY_OFF_FLAG),
                "snapshot": snap,
                "config": cfg,
                "gate3": gate3,
                "proactive": proactive,
                "models": self._chat_model_options(),
            }}
        except Exception as exc:
            return {"status": "error", "message": repr(exc)}

    async def web_console_post(self):
        try:
            from astrbot.api.web import request as _webreq
            body = await _webreq.json({}) or {}
            action = str(body.get("action") or "save_config")
            if action == "gate":
                off = bool(body.get("off"))
                if off:
                    with open(GROUP_REPLY_OFF_FLAG, "w", encoding="utf-8") as fh:
                        fh.write("off")
                elif os.path.exists(GROUP_REPLY_OFF_FLAG):
                    os.remove(GROUP_REPLY_OFF_FLAG)
                logger.info("[zhiyu_toolbox] 控制台总闸开关: " + ("关闭" if off else "开启"))
                return {"status": "ok", "data": {"gate_off": off}}
            if action == "toggle":
                # 统一开关：按 key 写回真正生效的那份配置，切换后立即起效。
                key = str(body.get("key") or "").strip()
                val = bool(body.get("value"))
                if key == "gate":
                    # 勾选 = 允许群聊（删掉静默标记）
                    if val:
                        if os.path.exists(GROUP_REPLY_OFF_FLAG):
                            os.remove(GROUP_REPLY_OFF_FLAG)
                    else:
                        with open(GROUP_REPLY_OFF_FLAG, "w", encoding="utf-8") as fh:
                            fh.write("off")
                    logger.info("[zhiyu_toolbox] 统一开关 群聊总闸: " + ("开启" if val else "关闭"))
                    return {"status": "ok", "data": {"key": key, "value": val}}
                if key in ("send_rate_enabled", "group_gap_enabled"):
                    self.config[key] = val
                    self._persist_config()
                    logger.info("[zhiyu_toolbox] 统一开关 " + key + ": " + str(val))
                    return {"status": "ok", "data": {"key": key, "value": val}}
                if key in ("gate3_enabled", "proactive_reply_enabled"):
                    comp = self._companion_star()
                    ccfg = getattr(comp, "config", None)
                    if ccfg is None:
                        return {"status": "error", "message": "没找到主动陪伴插件配置，无法保存开关"}
                    ccfg[key] = val
                    try:
                        ccfg.save_config()
                    except Exception as exc:
                        logger.warning("[zhiyu_toolbox] 主动陪伴配置 save_config() 失败: " + repr(exc))
                    logger.info("[zhiyu_toolbox] 统一开关 " + key + ": " + str(val))
                    return {"status": "ok", "data": {"key": key, "value": val}}
                return {"status": "error", "message": "未知开关：" + key}
            if action == "save_probability":
                try:
                    p = float(body.get("value"))
                except (TypeError, ValueError):
                    return {"status": "error", "message": "概率必须是 0~1 的数字"}
                p = max(0.0, min(1.0, p))
                comp = self._companion_star()
                ccfg = getattr(comp, "config", None)
                if ccfg is None:
                    return {"status": "error", "message": "没找到主动陪伴插件配置，无法保存概率"}
                ccfg["proactive_reply_probability"] = p
                try:
                    ccfg.save_config()
                except Exception as exc:
                    logger.warning("[zhiyu_toolbox] 主动陪伴配置 save_config() 失败: " + repr(exc))
                logger.info("[zhiyu_toolbox] 控制台设置主动回复概率: " + str(p))
                return {"status": "ok", "data": {"probability": p}}
            if action == "save_gate3":
                pid = str(body.get("provider_id") or "").strip()
                if pid not in ("", "@current") and pid not in {m["id"] for m in self._chat_model_options()}:
                    return {"status": "error", "message": "未知模型，未保存：" + pid}
                comp = self._companion_star()
                if comp is None:
                    return {"status": "error", "message": "没找到主动陪伴插件，无法保存裁判模型"}
                ccfg = getattr(comp, "config", None)
                if ccfg is None:
                    return {"status": "error", "message": "主动陪伴插件没有可写配置对象"}
                ccfg["gate3_provider_id"] = pid
                try:
                    ccfg.save_config()
                except Exception as exc:
                    logger.warning("[zhiyu_toolbox] 主动陪伴配置 save_config() 失败: " + repr(exc))
                logger.info("[zhiyu_toolbox] 控制台设置第三关裁判模型: " + (pid or "(留空=不判定)"))
                return {"status": "ok", "data": {"provider_id": pid}}
            patch = body.get("config") or {}
            applied = {}
            for key, val in patch.items():
                if key not in CONSOLE_CONFIG_KEYS:
                    continue
                newv = _console_sanitize(key, val)
                if newv is not None:
                    self.config[key] = newv
                    applied[key] = newv
            self._persist_config()
            return {"status": "ok", "data": {"saved": True, "applied": applied}}
        except Exception as exc:
            return {"status": "error", "message": repr(exc)}

    @filter.event_message_type(filter.EventMessageType.ALL, priority=300000)
    async def direct_status_card(self, event: AstrMessageEvent):
        # 管理员明确问运行状态时直接发卡并终止事件。
        # priority 300000 高于陪伴插件 240000、记忆插件 1000，
        # 所以不会进模型、不写会话历史、不会被任何记忆采集钩子记录。
        try:
            raw = str(event.get_message_str() or "")
            text = re.sub(r"\[.*?\]", "", raw).strip()
            if not text or len(text) > 40:
                return
            if not STATUS_DIRECT_RE.search(text):
                return
            ok, _ = self._check(event)
            if not ok:
                # 非管理员命中查状态问法：静默不理，不发任何内容、不进模型、不进记忆
                event.stop_event()
                logger.info("[zhiyu_toolbox] 非管理员查状态：静默不理，已跳过模型与记忆")
                return
            data = collect()
            img = None
            try:
                img = render_status_image(data)
            except Exception as exc:
                logger.warning("[zhiyu_toolbox] 状态图渲染失败，回退文本卡: " + repr(exc))
            if img:
                try:
                    from astrbot.api.message_components import Image
                    await event.send(event.chain_result([Image.fromBytes(img)]))
                except Exception as exc:
                    logger.warning("[zhiyu_toolbox] 图片发送失败，回退文本卡: " + repr(exc))
                    await event.send(event.plain_result(format_card(data)))
            else:
                await event.send(event.plain_result(format_card(data)))
            event.stop_event()
            logger.info("[zhiyu_toolbox] 管理员查状态：直出卡片，已跳过模型与记忆")
        except Exception as exc:
            logger.warning("[zhiyu_toolbox] 直出状态卡失败: " + repr(exc))


    # ------------------------------------------------- 第三个功能：发言清洗
    def _strip_echo(self, text: str, event: AstrMessageEvent) -> tuple:
        """去掉开头照抄对方原话的部分。

        模型常把对方的问题先抄一遍再回答（"用的什么技术？ 插件自带的…"）。
        群聊里的 message_str 往往带发送者前缀（"酒儿: …"）、把多条消息合并、
        或者夹着引用块，所以不能整段比开头。这里按换行、冒号、引用标记切成
        多个候选，逐段和回复开头比对，命中 6 个字以上就认定是复述，
        把这段删掉只留回答；删完太短就保留原样，免得整条消失。
        """
        # 模型偶尔会自造 &&fool&& 这类标记当前缀，先清掉再比对
        text = re.sub(r"&&[a-zA-Z_]{1,20}&&", "", text or "").strip()
        if not text:
            return "", "整条是标记不发送"
        try:
            src = event.get_message_str() or ""
        except Exception:
            return text, ""
        src = re.sub(r"<Quoted Message>.*?</Quoted Message>", " ", src, flags=re.S)
        src = re.sub(r"<system_reminder>.*?</system_reminder>", " ", src, flags=re.S)
        src = src.replace("[引用消息]", " ")

        def norm(s):
            return re.sub(r"[^\w\u4e00-\u9fff]", "", s or "")

        b = norm(text)
        if len(b) < 4:
            return text, ""

        candidates = [
            norm(p)
            for p in re.split(r"[\r\n]+|：|:|\[引用消息\]", src)
            if p and p.strip()
        ]
        # 整条就是对方原话（只改了标点、少个"你"、多了个表情）——不发
        for a in candidates:
            if len(a) >= 4 and len(b) <= len(a) + 3:
                if difflib.SequenceMatcher(None, b, a).ratio() >= 0.85:
                    return "", "整条是复述不发送"

        # 整条是括号旁白（自己嘀咕"我不接话了"，不该发到群里）
        if re.fullmatch(r"[（(][^）)]{1,}[）)]", text.strip()):
            return "", "括号旁白不发送"

        # [2026-11-11 复读治理] 半截括号旁白：模型把「谁刚说了啥、群里在聊啥」
        # 写成很长的（听着… /（看着… 旁白后被长度限制切断，只剩复述原话+背景，
        # 自己要说的那句没了。上面那条只认闭合括号，这里补上未闭合的。
        _s = text.strip()
        if _s.startswith(("（", "(")) and "）" not in _s and ")" not in _s:
            return "", "半截括号旁白不发送"

        # 闭合的「情境复述」旁白：剥掉括号段，只留后面真正要说的那句
        _m = re.match(r"^[（(]([^）)]{1,})[）)](.*)$", _s, flags=re.S)
        if _m:
            _inner, _rest = _m.group(1), _m.group(2).strip()
            if ("听着" in _inner or "看着" in _inner or "瞧着" in _inner) and (
                "结合前面" in _inner or "群里在聊" in _inner or "刚才" in _inner
            ):
                if len(norm(_rest)) < 4:
                    return "", "整条是情境复述不发送"
                return _rest, "删情境旁白"

        best = 0
        whole = False
        for a in candidates:
            n = 0
            limit = min(len(a), len(b), 24)
            while n < limit and a[n] == b[n]:
                n += 1
            if n > best:
                best = n
                # 对方这一整句都被抄走了（短句也算），单独记一笔
                whole = n == len(a) and n >= 4
        if best < 6 and not whole:
            return text, ""

        cnt = 0
        pos = 0
        visible = re.compile(r"[^\w\u4e00-\u9fff]")
        for i, ch in enumerate(text):
            if visible.match(ch):
                continue
            cnt += 1
            if cnt >= best:
                pos = i + 1
                break
        rest = text[pos:].lstrip("，,、；;：: \u3000？?！!。.～~")
        if len(norm(rest)) < 4:
            # 整句被抄走、删完只剩表情：这条没有任何自己的内容，不发
            if whole and best >= 4:
                return "", "整条是复述不发送"
            return text, ""
        return rest, f"删复述{best}字"


    # ---------------------------------------------------------------- 拼贴复述检测
    _ECHO_KEEP = 5

    def _remember_text(self, bucket: str, key: str, text: str) -> None:
        """把一条原话记进环形队列，只留最近 _ECHO_KEEP 条。"""
        if not text or not text.strip():
            return
        store = getattr(self, bucket, None)
        if not isinstance(store, dict):
            store = {}
            setattr(self, bucket, store)
        q = store.setdefault(key, [])
        if q and q[-1] == text:
            return
        q.append(text)
        del q[:-self._ECHO_KEEP]

    @staticmethod
    def _norm_echo(s: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", s or "")

    def _is_copy_piece(self, piece: str, sources, ratio: float = 0.8) -> bool:
        """这一小句的 3 字片段有多少能在某条原话里找到；够多就是搬来的。"""
        s = self._norm_echo(piece)
        if len(s) < 6:
            return False
        grams = {s[i:i + 3] for i in range(len(s) - 2)}
        if not grams:
            return False
        for src in sources:
            t = self._norm_echo(src)
            if len(t) < 4:
                continue
            tg = {t[i:i + 3] for i in range(len(t) - 2)}
            if len(grams & tg) / len(grams) >= ratio:
                return True
        return False

    def _strip_assembled(self, text: str, event, own_sources) -> tuple:
        """把回复按标点切块，逐块查是不是整段搬来的。

        比对三个地方：对方这一条、对方最近几条、自己最近几条。
        全都被判成搬来的就整条不发；只删掉搬来的那几块，剩下的照常发。
        """
        try:
            key = str(event.get_group_id() or "") or "private"
        except Exception:
            key = "private"
        srcs = list((getattr(self, "_recent_src_text", None) or {}).get(key, []))
        try:
            cur = event.get_message_str() or ""
            cur = re.sub(r"<Quoted Message>.*?</Quoted Message>", " ", cur, flags=re.S)
            cur = re.sub(r"<system_reminder>.*?</system_reminder>", " ", cur, flags=re.S)
            if cur.strip():
                srcs.append(cur)
        except Exception:
            pass
        if own_sources:
            srcs.extend(own_sources)

        segments = re.findall(r"[^，。！？、；\n]+[，。！？、；\n]?", text)
        if len(segments) <= 1:
            return text, ""

        kept, dropped = [], 0
        for seg in segments:
            body = seg.strip(" ，,、；;。！？!?\n")
            if len(self._norm_echo(body)) >= 6 and self._is_copy_piece(body, srcs):
                dropped += len(self._norm_echo(body))
                continue
            kept.append(seg)
        result = "".join(kept).strip(" ，,、；;。！？!?\n")
        if not result or len(self._norm_echo(result)) < 4:
            return "", "整条是拼贴不发送"
        if dropped:
            return result, "删拼贴%d字" % dropped
        return text, ""

    @filter.on_decorating_result(priority=-2000000)
    async def clean_outgoing(self, event: AstrMessageEvent):
        """发送前最后一道：清掉占位/括号动作/空行，压波浪号。

        优先级设为最小，保证在陪伴插件、表情包插件都处理完之后，对最终文本动手。
        整条只剩占位时清空消息链——框架遇到空消息链会直接跳过发送。
        """
        try:
            if not bool(self.config.get("clean_enabled", True)):
                return
            scope = str(self.config.get("clean_scope") or "all")
            group = str(event.get_group_id() or "")
            if scope == "group" and not group:
                return
            if scope == "private" and group:
                return
            result = event.get_result()
            if result is None:
                return
            rct = getattr(result, "result_content_type", None)
            if rct is not None and "STREAMING" in str(rct):
                return  # 流式输出不碰，避免破坏分片
            chain = getattr(result, "chain", None)
            if not isinstance(chain, list) or not chain:
                return
            try:
                from astrbot.api.message_components import Plain
                is_plain = lambda comp: isinstance(comp, Plain)
            except ImportError:
                is_plain = lambda comp: type(comp).__name__ == "Plain"

            chat_key = str(event.get_group_id() or "") or "private"
            try:
                self._remember_text("_recent_src_text", chat_key, event.get_message_str() or "")
            except Exception:
                pass
            actions = []
            base_parts = []
            for comp in chain:
                if not is_plain(comp):
                    continue
                new_text, acts = clean_outgoing_text(getattr(comp, "text", ""), self.config)
                actions.extend(acts)
                if new_text:
                    new_text, echo_act = self._strip_echo(new_text, event)
                    if echo_act:
                        actions.append(echo_act)
                    if new_text:
                        new_text, asm_act = self._strip_assembled(
                            new_text, event, (getattr(self, "_recent_own_text", None) or {}).get(chat_key, [])
                        )
                        if asm_act:
                            actions.append(asm_act)
                    base_parts.append(new_text)  # 用作去重键
                comp.text = new_text

            left = "".join(
                getattr(c, "text", "").strip() for c in chain if is_plain(c)
            )
            has_media = any(
                type(c).__name__ not in ("Plain", "Reply", "At") for c in chain
            )
            # 发送节流：每分钟条数限制与单群插话间隔各自独立开关，互不影响。
            # 被@的回复始终放行，不计入任一限制。
            if (left or has_media) and chain:
                rate_on = bool(self.config.get("send_rate_enabled", True))
                gap_on = bool(self.config.get("group_gap_enabled", True))
                win = _clean_int(self.config, "send_rate_window", 60, 1, 3600)
                cap = _clean_int(self.config, "send_rate_max", 4, 1, 999)
                is_at_reply = False
                try:
                    self_id = str(event.get_self_id() or "")
                    msgs = getattr(getattr(event, "message_obj", None), "message", None) or []
                    for mc in msgs:
                        if type(mc).__name__ == "At" and str(getattr(mc, "qq", "")) == self_id:
                            is_at_reply = True
                            break
                except Exception:
                    is_at_reply = False
                if is_at_reply:
                    actions.append("被@不计数")
                else:
                    now = time.monotonic()
                    gid = str(event.get_group_id() or "")
                    dropped = False
                    # ① 单群插话间隔：独立开关，关掉就完全不等待
                    if gap_on:
                        gmin = _clean_int(self.config, "group_gap_min_seconds", 0, 0, 3600)
                        gmax = _clean_int(self.config, "group_gap_max_seconds", 0, 0, 3600)
                        if gmax < gmin:
                            gmax = gmin
                        gap = random.randint(gmin, gmax) if (gmin and gid) else 0
                        lastg = getattr(self, "_last_gap_send", None)
                        if lastg is None:
                            lastg = {}
                            self._last_gap_send = lastg
                        if gap and gid and now - lastg.get(gid, 0.0) < gap:
                            chain.clear()
                            left = ""
                            actions.append("单群间隔未到丢弃")
                            dropped = True
                        elif gap and gid:
                            lastg[gid] = now
                    # ② 每分钟条数限制：独立开关，与间隔互不依赖
                    if not dropped and rate_on and win and cap:
                        q = getattr(self, "_send_times", None)
                        if q is None:
                            q = []
                            self._send_times = q
                        while q and now - q[0] > win:
                            q.pop(0)
                        if len(q) >= cap:
                            chain.clear()
                            left = ""
                            actions.append("插话超速丢弃")
                        else:
                            q.append(now)
            if not left and not has_media:
                chain.clear()
                actions.append("整条不发送")
            elif left:


                # 短时间内重复同一句话就不发：群里消息密集时模型容易复读自己上一句。
                # 按群分开记，带图片的消息也照样比对文字；比较前去掉表情、标点和波浪号，
                # 免得模型加个 🐾 或改个句号就把同一句当成新的。
                dedupe = _clean_int(self.config, "clean_dedupe_seconds", 90, 0, 600)
                if dedupe:
                    now = time.monotonic()
                    scope = str(event.get_group_id() or "") or "private"
                    store = getattr(self, "_recent_sent_by_scope", None) or {}
                    recent = [
                        item for item in store.get(scope, [])
                        if now - item[0] < dedupe
                    ]
                    key = _dedupe_key("".join(base_parts) or left)
                    if key and any(item[1] == key for item in recent):
                        chain.clear()
                        actions.append("重复句不发送")
                    else:
                        if key:
                            recent.append((now, key))
                        store[scope] = recent[-10:]
                        self._recent_sent_by_scope = store
            if left and chain:
                try:
                    self._remember_text("_recent_own_text", chat_key, left)
                except Exception:
                    pass

            if actions and bool(self.config.get("clean_verbose_log", False)):
                logger.info(f"[zhiyu_toolbox] 发言清洗：{'/'.join(actions)}")
        except Exception as exc:
            # 清洗是旁路，任何异常都不能影响正常发消息
            logger.warning(f"[zhiyu_toolbox] 发言清洗异常（已忽略）: {exc!r}")

    # ------------------------------------------------------------ 扩展区
    # 【以后要加新功能，全部加在这个类里，不用新建插件、不用重装、不用动旧功能】
    #
    # 步骤：
    #   1. 需要读文件/配置/数据库，就在上面的「采集函数区」加一个 _ 开头的函数，
    #      复用 _load_json / CFG_FILE / DB_FILE，别在工具方法里堆逻辑。
    #   2. 在本类里加一个 @filter.llm_tool(name="zhiyu_功能名") 的 async 方法；
    #      docstring 写清楚「听到什么样的话就该调它」，模型靠这段描述自己辨认，
    #      不需要斜杠命令，也不需要固定指令词。
    #   3. 方法第一行先 ok, reason = self._check(event) 做硬校验；涉及运行数据、
    #      用量、配置的功能，照抄 query_status 里「权限未通过」那段返回话术。
    #      纯娱乐、不含数据的功能才可以用 self._check(event, scope="everyone")。
    #   4. 结果用 await event.send(event.plain_result(文本)) 直接发，
    #      再 return 一句给模型的短提示，别让模型复述或编数字。
    #   5. 新功能要单独开关，就在 _conf_schema.json 里加一个新键，
    #      用 self.config.get("新键", 默认值) 读；旧键名一律不许改，
    #      否则面板里已保存的设置会丢。
    #
    # 模板（复制后改工具名、docstring 和采集函数即可）：
    #   @filter.llm_tool(name="zhiyu_xxx")
    #   async def xxx(self, event: AstrMessageEvent):
    #       """当用户问「……」时调用本工具，用来做……。"""
    #       ok, reason = self._check(event)
    #       if not ok:
    #           return f"权限校验未通过（{reason}）。只回一句很短、带点歉意的话，不要解释原因。"
    #       text = "……"   # 由采集函数区的新函数生成
    #       await event.send(event.plain_result(text))
    #       return "结果已直接发出，只回一句很短的话，不要复述内容。"
