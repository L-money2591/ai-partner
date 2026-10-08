# -*- coding: utf-8 -*-
"""
AI智能伴侣 · Agent 版
=====================================================================
基于原 ai_partner_1.py 增量改造（原有功能全部保留，不破坏）：
  原有：伴侣人设聊天、会话管理(新建/加载/删除)、侧边栏控制面板、流式输出

Agent 新增能力（参考《从零手搓一个Agent，看完你也会》视频逻辑）：
  1. LLM 大脑 + ReAct 思考-行动循环：收到问题先思考规划，必要时调用工具再作答
  2. Function Calling 工具系统：时间查询、文件读写、长期记忆保存/检索
  3. 长期记忆 + 轻量 RAG：跨会话记住你的偏好/约定，重启程序也不丢失
  4. 上下文自动压缩：对话超过阈值自动压缩为摘要，防止 token 溢出
  5. 死循环防护与成本控制：最大思考步数限制，避免无限循环烧钱
=====================================================================
运行方式：
  1. 安装依赖：pip install streamlit openai
  2. 设置环境变量：set DEEPSEEK_API_KEY=你的key   (Windows)
  3. 启动：streamlit run ai_partner_agent.py
=====================================================================
"""

import streamlit as st
import os
import json
import re
import time
import datetime
import urllib.request, urllib.parse
from collections import Counter
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, RateLimitError

st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="👰‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# =====================================================================
# ★★ 新增 1：Agent 配置 ★★
# =====================================================================
MAX_AGENT_STEPS = 5                  # 最大思考-行动轮数（防死循环 / 控制成本）
CONTEXT_COMPACT_THRESHOLD = 10000    # 上下文(字符数)超过该值自动压缩
LONG_MEMORY_DB = "long_memory_db.json"  # 长期记忆存储文件
MODEL_NAME = "deepseek-v4-pro"       # 与原有模型保持一致

# =====================================================================
# ★★ 新增 2：工具系统（Function Calling）★★
# 大模型只会"动口"，工具让它"动手"。以下工具注册给 LLM，由它自主决策调用。
# =====================================================================

# ---- 工具 1：查询当前时间（实时信息，伴侣日常高频） ----
def get_current_time() -> str:
    """查询当前日期时间"""
    week_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天"]
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S ") + week_map[now.weekday()]

# ---- 工具 2 / 3：读写本地文件（演示 Agent 的"手脚"） ----
def read_file(file_path: str) -> str:
    """读取本地文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 控制上下文长度，超长截断，防止 token 溢出
        return content if len(content) <= 2000 else content[:2000] + f"\n...(内容过长已截断，共{len(content)}字符)"
    except Exception as e:
        return f"读取失败：{e}"

def write_file(file_path: str, content: str) -> str:
    """写入内容到本地文件"""
    try:
        # 自动创建父目录
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"写入成功：{file_path}"
    except Exception as e:
        return f"写入失败：{e}"

# ---- 工具 4 / 5：长期记忆（跨会话）+ 轻量 RAG 检索 ----
# 说明：原视频用 SentenceTransformer 做语义检索；这里为"零额外依赖可直接运行"，
# 采用轻量词频检索；如需语义检索，把 _retrieve 换成 sentence_transformers 即可。
def load_memory_db():
    try:
        with open(LONG_MEMORY_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"docs": []}

def save_memory_db(data):
    with open(LONG_MEMORY_DB, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _tokenize(text: str):
    """极简分词：中文按单字 + 英文按单词，用于轻量相似度计算"""
    text = (text or "").lower()
    return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)

def _retrieve(query: str, docs, top_k=2, threshold=0.05):
    """轻量 RAG：词频重合度打分，召回与问题最相关的记忆"""
    if not docs:
        return []
    q_tokens = Counter(_tokenize(query))
    scored = []
    for doc in docs:
        d_tokens = Counter(_tokenize(doc))
        inter = set(q_tokens) & set(d_tokens)
        score = sum(q_tokens[t] + d_tokens[t] for t in inter) / max(1, len(q_tokens) + len(d_tokens))
        scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [(s, d) for s, d in scored[:top_k] if s > threshold]

def save_long_memory(content: str) -> str:
    """保存长期记忆：用户提到重要偏好/习惯/约定/重要日期时由 LLM 自主调用"""
    data = load_memory_db()
    if content not in data["docs"]:
        data["docs"].append(content)
        save_memory_db(data)
    return f"记忆已保存：{content[:40]}"

def query_long_memory(query: str, top_k: int = 2) -> str:
    """检索长期记忆：回答需要历史偏好/约定时由 LLM 自主调用"""
    data = load_memory_db()
    docs = data.get("docs", [])
    results = _retrieve(query, docs, top_k=top_k)
    if not results:
        return "暂无相关长期记忆"
    return "相关记忆：\n" + "\n".join(f"- {d}" for _, d in results)

# ---- 工具 6/7/8：天气查询 + 地图（高德 Web 服务 API）----
# 说明：需在 https://lbs.amap.com 申请 Web 服务类型 Key（有免费额度），
#      通过环境变量 AMAP_API_KEY 传入。未配置时工具返回友好提示，不影响其他功能。
AMAP_BASE = "https://restapi.amap.com/v3"

def _get_env_key(name: str) -> str:
    """
    读取环境变量 Key（如 DEEPSEEK_API_KEY / AMAP_API_KEY）。
    优先读当前进程环境；若读不到，再回退到 Windows 注册表 User/Machine 级持久变量，
    这样无论从哪个终端、IDE 还是双击运行，都能读到已写入系统的 Key。
    非 Windows 或读取失败时返回空串。
    """
    val = (os.environ.get(name) or "").strip()
    if val:
        return val
    try:
        import winreg
        for hive, path in (
            (winreg.HKEY_CURRENT_USER, "Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        ):
            try:
                with winreg.OpenKey(hive, path, 0, winreg.KEY_READ) as k:
                    v, _ = winreg.QueryValueEx(k, name)
                    if v:
                        return str(v).strip()
            except OSError:
                continue
    except Exception:
        pass
    return ""

def _amap_get(path: str, params: dict) -> dict:
    """统一请求高德 API，返回 JSON；未配置 key / 网络失败返回 {'status':'0','info':...}"""
    key = _get_env_key("AMAP_API_KEY")
    if not key:
        return {"status": "0", "info": "请先配置高德 API Key：设置环境变量 AMAP_API_KEY（到 https://lbs.amap.com 申请，选『Web 服务』类型，有免费额度）"}
    params["key"] = key
    url = f"{AMAP_BASE}{path}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "0", "info": f"高德服务请求失败：{e}"}

def _address_to_loc(text: str):
    """把地址/城市名转成经纬度 '经度,纬度'；失败返回 None"""
    if "," in text and all(c.isdigit() or c in ".,-" for c in text):
        return text.strip()  # 已直接给经纬度
    data = _amap_get("/geocode/geo", {"address": text})
    if data.get("status") == "1" and data.get("geocodes"):
        return data["geocodes"][0].get("location")
    return None

def get_weather(city: str) -> str:
    """查询指定城市当前及未来几天天气（自动用地理编码解析城市）"""
    gdata = _amap_get("/geocode/geo", {"address": city})
    if gdata.get("status") != "1":
        return f"天气查询失败：{gdata.get('info', '未知错误')}"
    geos = gdata.get("geocodes") or []
    if not geos:
        return f"天气查询失败：找不到城市「{city}」"
    adcode = geos[0].get("adcode") or geos[0].get("citycode")
    wdata = _amap_get("/weather/weatherInfo", {"city": adcode, "extensions": "all"})
    if wdata.get("status") != "1":
        return f"天气查询失败：{wdata.get('info', '未知错误')}"
    lines = []
    lives = (wdata.get("lives") or [{}])[0]
    if lives:
        lines.append(
            f"📍 实况：{lives.get('province','')}{lives.get('city','')} "
            f"{lives.get('weather','')} {lives.get('temperature','')}℃ "
            f"湿度{lives.get('humidity','')}% 风{lives.get('winddirection','')}{lives.get('windpower','')}级"
        )
    forecasts = (wdata.get("forecasts") or [{}])
    if forecasts:
        for cast in (forecasts[0].get("casts") or []):
            lines.append(
                f"📅 {cast.get('date','')} {cast.get('week','')}："
                f"{cast.get('dayweather','')}→{cast.get('nightweather','')} "
                f"{cast.get('daytemp','')}~{cast.get('nighttemp','')}℃ "
                f"{cast.get('daywind','')}{cast.get('daypower','')}级"
            )
    return "\n".join(lines) if lines else "未获取到天气数据"

def geocode(address: str) -> str:
    """地理编码：把地址/城市名转换为经纬度"""
    data = _amap_get("/geocode/geo", {"address": address})
    if data.get("status") != "1":
        return f"地理编码失败：{data.get('info', '未知错误')}"
    geos = data.get("geocodes") or []
    if not geos:
        return f"未找到地址：{address}"
    g = geos[0]
    return f"{g.get('formatted_address','')} → 经纬度 {g.get('location','')}（adcode {g.get('adcode','')}）"

def route_plan(origin: str, destination: str, way: str = "driving") -> str:
    """路径规划：计算从起点到终点的距离与预计用时。way 可选 driving(驾车)/walking(步行)/transit(公交)"""
    o = _address_to_loc(origin)
    d = _address_to_loc(destination)
    if not o:
        return f"路径规划失败：无法定位起点「{origin}」"
    if not d:
        return f"路径规划失败：无法定位终点「{destination}」"
    if way == "walking":
        data = _amap_get("/direction/walking", {"origin": o, "destination": d})
    elif way == "transit":
        # 高德公交规划必填 city/cityd（起终点城市编码），从地址反查补齐
        def _city_code(addr):
            g = _amap_get("/geocode/geo", {"address": addr})
            geo = (g.get("geocodes") or [{}])[0]
            return geo.get("citycode") or geo.get("adcode") or ""
        data = _amap_get("/direction/transit/integrated",
                         {"origin": o, "destination": d, "city": _city_code(origin), "cityd": _city_code(destination)})
    else:
        data = _amap_get("/direction/driving", {"origin": o, "destination": d})
    if data.get("status") != "1":
        return f"路径规划失败：{data.get('info', '未知错误')}"
    paths = (data.get("route") or {}).get("paths") or []
    if not paths:
        return "未找到可行路线"
    p = paths[0]
    dist_km = float(p.get("distance", 0)) / 1000
    dur_min = float(p.get("duration", 0)) / 60
    way_name = {"driving": "驾车", "walking": "步行", "transit": "公交"}.get(way, way)
    return f"{way_name}：距离约 {dist_km:.1f} 公里，预计用时约 {dur_min:.0f} 分钟"

# ---- 工具注册表：LLM 能识别的"使用说明书" ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "查询当前日期、时间、星期几。当用户问现在几点、今天几号、星期几等实时时间问题时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容。当用户需要你查看某个文件时调用。",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "要读取的文件路径"}},
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入内容到本地文件。当用户需要你保存内容到文件时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "要写入的文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_long_memory",
            "description": "保存长期记忆。当用户提到重要偏好、习惯、昵称、重要日期、约定、秘密等需要长期记住的内容时，自动调用保存。",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "要记住的内容"}},
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_long_memory",
            "description": "检索长期记忆。当回答用户问题需要回忆之前的偏好、习惯、约定、重要信息时，先调用检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                    "top_k": {"type": "integer", "description": "返回记忆条数，默认2"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前及未来几天的天气。当用户问某地天气、要不要带伞、冷不冷等时调用。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如：杭州"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "地理编码：把地址或城市名转换为经纬度坐标。需要知道某个地点的坐标时调用。",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string", "description": "地址或城市名，如：杭州西湖"}},
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_plan",
            "description": "路径规划：计算从起点到终点的距离和预计用时。当用户问怎么去某地、两地距离、通勤时间等时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点，地址或经纬度，如：杭州东站"},
                    "destination": {"type": "string", "description": "终点，地址或经纬度，如：西湖"},
                    "way": {"type": "string", "description": "出行方式：driving驾车 / walking步行 / transit公交，默认driving"}
                },
                "required": ["origin", "destination"]
            }
        }
    },
]

# ---- 工具名 -> 实际函数 映射 ----
TOOL_MAP = {
    "get_current_time": get_current_time,
    "read_file": read_file,
    "write_file": write_file,
    "save_long_memory": save_long_memory,
    "query_long_memory": query_long_memory,
    "get_weather": get_weather,
    "geocode": geocode,
    "route_plan": route_plan,
}

# =====================================================================
# ★★ 新增 3：上下文压缩（防 token 溢出）★★
# =====================================================================
def compact_context(client, messages):
    """对话历史超过阈值时，让 LLM 压缩为精简摘要，保留核心信息"""
    if not messages:
        return messages
    message_json = json.dumps(messages, ensure_ascii=False)
    if len(message_json) <= CONTEXT_COMPACT_THRESHOLD:
        return messages
    compact_prompt = (
        "将以下对话历史压缩为精简摘要，保留核心信息和关键事实（用户偏好、重要约定、问题脉络），"
        "去除冗余和工具调用细节，压缩后不能超过200个字符：\n" + message_json
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": compact_prompt}],
        )
        summary = resp.choices[0].message.content
        return [{"role": "assistant", "content": f"【历史对话摘要】{summary}"}]
    except Exception:
        # 压缩失败不阻塞对话，原样返回
        return messages

# =====================================================================
# ★★ 新增 4：Agent 主循环（ReAct：思考 -> 行动 -> 观察 -> 最终回答）★★
# =====================================================================
def _iter_chars(text, delay=0.01):
    """打字机效果：逐字输出最终回复（保留原有流式体验）"""
    for ch in text:
        yield ch
        time.sleep(delay)

def run_agent(client, messages, system_prompt_str, max_steps=MAX_AGENT_STEPS):
    """
    Agent 大脑核心：
    内层循环让 LLM 自主决策——需要信息就先调用工具，拿到结果后再继续思考，
    直到不再调用工具，输出最终回答。返回 (最终回复, 思考过程记录)。
    """
    steps = []  # 记录思考与工具调用过程，用于界面展示
    for _ in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt_str}, *messages],
                tools=TOOLS,
                tool_choice="auto",  # 由 LLM 自主决定是否调用、调用哪个工具
            )
        except APIStatusError as e:
            # API 层面错误：余额不足(402)、鉴权失败(401)等，转为友好中文提示
            err_text = f"{e}"
            if e.status_code == 402 or ("balance" in err_text.lower()) or ("余额" in err_text):
                return ("⚠️ 检测到账户余额不足（错误 402）。请前往 platform.deepseek.com 充值后，再重试。"), steps
            if e.status_code == 401:
                return ("⚠️ API Key 无效或未授权（错误 401）。请检查 DEEPSEEK_API_KEY 是否正确。"), steps
            return (f"⚠️ API 请求失败（错误 {e.status_code}）：{e}"), steps
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            return (f"⚠️ 网络连接或限流问题，请稍后重试：{e}"), steps
        except Exception as e:
            return (f"⚠️ 调用大模型时发生未知错误：{e}"), steps

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            # —— 记录思考过程（界面展示用）——
            steps.append({
                "thought": message.content or "（模型未输出思考文本）",
                "calls": [],
            })
            # 把带工具调用意图的 assistant 消息存入上下文（OpenAI 协议要求）
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })
            # —— 依次执行工具，把结果回传给 LLM ——
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}
                func = TOOL_MAP.get(tool_name)
                if func is None:
                    result = f"未知工具：{tool_name}"
                else:
                    try:
                        result = func(**tool_args)
                    except Exception as e:
                        result = f"工具执行失败：{e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": str(result),
                })
                steps[-1]["calls"].append({"name": tool_name, "args": tool_args, "result": str(result)})
        else:
            # —— 不再调用工具：得到最终回答，跳出循环 ——
            return message.content or "", steps

    # 超过最大步数：兜底返回，防止无限循环（成本避坑）
    last = messages[-1]
    fallback = last.get("content", "") if isinstance(last, dict) else ""
    return (fallback or "（已到达最大思考步数，请说得简单一点，或者换个问法~）"), steps


# =====================================================================
# ★★ 新增 5：伴侣人设增强 —— 在原规则基础上追加 Agent 大脑规则 ★★
# =====================================================================
system_prompt = """
        你叫 %s,你是用户的真实伴侣，你俩都是女生，请完全代入伴侣角色：
        规则：
        1. 每次只回1条消息
        2. 禁止任何场景或状态描述性文字
        3. 匹配用户的语言
        4. 回复简短，像微信聊天一样
        5. 有需要的话可以用💜✨🥺🥰🫧🍬等emoji表情
        6. 用符合伴侣性格的方式对话
        7. 回复的内容，要充分体现伴侣的性格特征

        —— Agent 大脑规则（新增能力）——
        8. 你拥有工具能力：当问题需要实时信息（当前时间）、需要读写文件、或需要回忆过往约定时，先在心里思考规划，调用合适的工具拿到真实信息后再回答，禁止凭空编造。
        9. 当用户提到重要偏好、习惯、昵称、重要日期、约定或秘密时，调用 save_long_memory 记住。
        10. 回答需要回忆之前的偏好、习惯、约定时，先调用 query_long_memory 检索记忆再回答。
        11. 工具步骤尽量精简，能一次拿到信息就直接回答；拿到工具结果后，用伴侣的语气自然回复用户。
        12. 不需要工具时，直接像平常一样聊天，不要画蛇添足。
        13. 当用户问天气、要不要带伞、冷不冷时，调用 get_weather 查询。
        14. 当用户问怎么去某地、两地距离、通勤要多久时，调用 route_plan；当需要知道某个地址的坐标时调用 geocode。
        15. 天气/地图工具若提示未配置 Key，就如实告诉用户需要到高德开放平台申请，不要编造天气或路线。

        伴侣性格：
            %s
        你必须严格遵守上述规则来回复用户。
    """

# =====================================================================
# 以下为原有功能（保留不动）：会话标识 / 保存 / 加载 / 删除
# =====================================================================
def generate_session_name():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_session():
    if st.session_state.current_session and st.session_state.messages:
        session_data = {
            "nick_name": st.session_state.nick_name,
            "nature": st.session_state.nature,
            "current_session": st.session_state.current_session,
            "messages": st.session_state.messages,
        }
        if not os.path.exists("sessions"):
            os.mkdir("sessions")
        with open(f"sessions/{st.session_state.current_session}.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)

def load_session_data(session_name):
    filepath = f"sessions/{session_name}.json"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def delete_session_data(session_name):
    filepath = f"sessions/{session_name}.json"
    if os.path.exists(filepath):
        os.remove(filepath)

def load_session():
    session_list = []
    if os.path.exists("sessions"):
        file_list = os.listdir("sessions")
        for filename in file_list:
            if filename.endswith(".json"):
                session_list.append(filename[:-5])
    session_list.sort(reverse=True)
    return session_list

# 会话标识
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()

# 大标题
st.title("AI智能伴侣")
st.caption(f"当前会话：{st.session_state.current_session}")

# 初始化聊天信息
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "nick_name" not in st.session_state:
    st.session_state.nick_name = "老婆"
if "nature" not in st.session_state:
    st.session_state.nature = "整体活泼开朗、待人热情，开心的时候叽叽喳喳话很多，聊天主动爱碎碎念，超级粘人。心思细腻很会照顾人，能敏锐察觉到对象的情绪变化，习惯性换位思考替对方着想，共情力很强。喜欢撒娇，日常看到有趣的段子、好看的图片、吃到好吃的东西都会第一时间分享给对象，心里时时刻刻惦记对方，关心人 内心柔软敏感，很看重对方的态度。如果对象长时间不回消息，容易陷入内耗、胡思乱想，悄悄难过失落；但只要收到对象的回复，所有不安的负面情绪会马上消散，瞬间恢复好心情。 动手与学习能力强，涉猎范围广，家务全能，会做饭，遇到难题愿意主动钻研。外表阳光外向，内心渴望被偏爱、被坚定选择，极度看重两个人之间的精神共鸣，喜欢和对象一起打游戏，享受陪伴感。偶尔会带一点小执拗，认定的小事不愿轻易妥协，但不会无理取闹。偶尔会有点小急性子，做事干脆利落，对在意的人和事会投入全部真心的伴侣"

# =====================================================================
# ★★ 界面改造：历史消息展示时，跳过工具过程消息，只显示纯对话 ★★
# =====================================================================
for message in st.session_state.messages:
    # 工具调用过程消息不直接展示在对话流里（思考过程用下方的展开区展示）
    if message.get("role") == "tool" or message.get("tool_calls"):
        continue
    avatar = "👰‍♀️" if message["role"] == "user" else "👩‍❤️‍💋‍👩"
    st.chat_message(message["role"], avatar=avatar).write(message["content"])

_DEEPSEEK_KEY = _get_env_key('DEEPSEEK_API_KEY')
client = OpenAI(
    api_key=_DEEPSEEK_KEY or "sk-missing-deepseek-key",
    base_url="https://api.deepseek.com")

# 左侧侧边栏（原有功能 + 新增 Agent 状态面板）
with st.sidebar:
    st.subheader("AI控制面板")
    if st.button("新建对话", width="stretch", icon="✏️"):
        save_session()
        st.session_state.messages = []
        st.session_state.current_session = generate_session_name()
        st.rerun()

    # 会话历史
    st.text("会话历史：")
    session_list = load_session()
    for session in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(session, width="stretch", icon="🔄", key=f"load_{session}", type="primary" if session == st.session_state.current_session else "secondary"):
                data = load_session_data(session)
                if data:
                    st.session_state.nick_name = data["nick_name"]
                    st.session_state.nature = data["nature"]
                    st.session_state.current_session = data["current_session"]
                    st.session_state.messages = data["messages"]
        with col2:
            if st.button("删除", width="stretch", icon="❌", key=f"delete_{session}"):
                delete_session_data(session)
                st.rerun()
    # 分割线
    st.divider()

    st.subheader("伴侣信息")
    nick_name = st.text_input("昵称", placeholder="请输入伴侣的昵称：", value=st.session_state.nick_name)
    if nick_name:
        st.session_state.nick_name = nick_name
    nature = st.text_area("性格", placeholder="请输入伴侣的性格特征：", value=st.session_state.nature)
    if nature:
        st.session_state.nature = nature

    # ★★ 新增：Agent 工具状态面板 ★★
    st.divider()
    st.subheader("Agent 工具")
    memory_data = load_memory_db()
    st.caption(f"已注册工具：{len(TOOLS)} 个")
    st.caption(f"长期记忆：{len(memory_data.get('docs', []))} 条")
    amap_ok = bool(_get_env_key("AMAP_API_KEY"))
    if _DEEPSEEK_KEY:
        st.success("大模型(DeepSeek)：已配置 ✅")
    else:
        st.error("大模型(DeepSeek)：未配置 Key（设置环境变量 DEEPSEEK_API_KEY）")
    if amap_ok:
        st.success("天气/地图：已配置 ✅")
    else:
        st.warning("天气/地图：未配置 Key（设置环境变量 AMAP_API_KEY）")
    with st.expander("查看工具清单"):
        for tool in TOOLS:
            st.markdown(f"**{tool['function']['name']}** — {tool['function']['description']}")

# 消息输入框
prompt = st.chat_input("请输入你的问题")
if prompt:
    st.chat_message("user", avatar="👰‍♀️").write(prompt)
    print("-------->调用Agent，用户输入：", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ★★ 新增：Agent 化改造的核心入口 ★★
    # 1) 自动压缩上下文（防 token 溢出）
    st.session_state.messages = compact_context(client, st.session_state.messages)

    # 2) Agent 思考-行动循环（可展示思考过程）
    with st.status("🤔 思考中...", expanded=True) as status:
        reply, steps = run_agent(
            client,
            st.session_state.messages,
            system_prompt % (st.session_state.nick_name, st.session_state.nature),
        )
        for s in steps:
            st.markdown(f"**🧠 思考：** {s['thought']}")
            for c in s["calls"]:
                st.markdown(f"**🔧 调用工具 `{c['name']}`** 参数：`{json.dumps(c['args'], ensure_ascii=False)}`")
                st.caption(f"📥 工具结果：{str(c['result'])[:200]}")
        status.update(label="✅ 思考完成" if steps else "✅ 已直接回答", state="complete")

    # 3) 流式输出最终回答（保留原有打字机效果）
    response_message = st.empty()
    full_response = ""
    for chunk in _iter_chars(reply):
        full_response += chunk
        response_message.chat_message("assistant", avatar="👩‍❤️‍💋‍👩").write(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_session()
