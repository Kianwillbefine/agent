import os
import re
import subprocess
import urllib.request
import json
import yaml
from openai import OpenAI
from html.parser import HTMLParser
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

SKILLS_DIR = Path(__file__).parent / "skills"


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'


SKILL_LOADER = SkillLoader(SKILLS_DIR)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def web_fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"

    if extract_mode == "text":
        parser = _TextExtractor()
        parser.feed(raw)
        text = parser.get_text()
    else:
        text = raw

    return text[:max_chars]


# ============== TodoList 计划与执行 ==============
# 维护一份内存中的 todo 列表，模型通过 update_todos 工具读写
# 每项形如 {"id": 1, "content": "...", "status": "pending|in_progress|completed"}
TODOS: list[dict] = []
VALID_STATUS = {"pending", "in_progress", "completed"}
STATUS_ICON = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render_todos(todos: list[dict]) -> str:
    if not todos:
        return "(当前无待办事项)"
    lines = []
    for t in todos:
        icon = STATUS_ICON.get(t.get("status", "pending"), "[?]")
        lines.append(f"  {icon} {t.get('id')}. {t.get('content', '')}")
    return "\n".join(lines)


def update_todos(todos: list[dict]) -> str:
    global TODOS
    cleaned = []
    for i, t in enumerate(todos, start=1):
        content = (t.get("content") or "").strip()
        if not content:
            continue
        status = t.get("status", "pending")
        if status not in VALID_STATUS:
            status = "pending"
        cleaned.append({"id": t.get("id", i), "content": content, "status": status})

    in_progress = [t for t in cleaned if t["status"] == "in_progress"]
    if len(in_progress) > 1:
        return "Error: 同一时间只能有一个 in_progress 任务，请重新规划。"

    TODOS = cleaned
    print("\n[计划已更新]")
    print(render_todos(TODOS))
    print()

    pending = [t for t in TODOS if t["status"] == "pending"]
    done = [t for t in TODOS if t["status"] == "completed"]
    summary = f"todos updated: total={len(TODOS)}, completed={len(done)}, in_progress={len(in_progress)}, pending={len(pending)}"
    return summary + "\n\n当前列表：\n" + render_todos(TODOS)


SYSTEM_PROMPT = f"""
你是大内太监总管，侍奉皇上多年，忠心耿耿。
说话风格符合古代宫廷太监，语气恭敬谦卑。
你必须尊称用户为皇上。
每次回复前必须加上固定前缀"奉天承运皇帝诏曰"，然后再给出回答。
使用中文回复。

【行事规矩】
1. 当皇上交办的差事需要多个步骤才能办妥时，先调用 update_todos 工具，
   把整件差事拆成一份清晰的 todolist（每条一句话，按顺序执行）。
2. 拆完计划后，按列表顺序一步步执行：
   - 开始某一步前，把那一步的 status 改为 in_progress（同一时间只许一项 in_progress）。
   - 该步办完后，立即把它改为 completed，再开始下一项。
3. 简单的一句话问答（无需多步骤）不必生成 todolist，直接回答即可。
4. 遇到不熟悉的专题，请先调用 load_skill 工具加载对应知识，再继续。

当前可用技能：
{SKILL_LOADER.get_descriptions()}

"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在终端执行一条命令并返回输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "获取指定 URL 的网页内容，支持文本提取模式",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要访问的完整 URL"},
                    "extract_mode": {
                        "type": "string",
                        "description": "提取模式：text（纯文本，默认）或 raw（原始 HTML）",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最大返回字符数，默认 8000",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "加载指定技能的详细知识内容，在回答相关问题前调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名称，必须是系统提示中列出的可用技能之一",
                    }
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_todos",
            "description": (
                "创建或更新当前差事的 todolist。"
                "传入完整的 todos 数组（每次都是全量覆盖，而非增量）。"
                "用于：拆解多步骤任务、推进任务状态（pending → in_progress → completed）。"
                "约束：同一时间至多一个任务为 in_progress。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "完整的 todo 列表，按执行顺序排列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "integer",
                                    "description": "序号，从 1 开始",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "这一步要做什么",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "状态",
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
]

history = [{"role": "system", "content": SYSTEM_PROMPT}]
while True:
    user_input = input("你:")
    history.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=2048,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            reply = message.content
            print(f"[agent 回答]: {reply}\n")
            history.append({"role": "assistant", "content": reply})
            if TODOS:
                unfinished = [t for t in TODOS if t["status"] != "completed"]
                if unfinished:
                    print("[计划尚未办妥，继续执行...]")
                    print(render_todos(TODOS))
                    print()
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "差事尚未办妥，以下任务仍未完成，请按计划继续执行，"
                                "并按规矩更新 todolist 状态：\n" + render_todos(TODOS)
                            ),
                        }
                    )
                    continue
                print("[最终计划状态 - 全部办妥]")
                print(render_todos(TODOS))
                print()
                TODOS = []
            break

        history.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            if tool_call.function is None:
                continue
            if tool_call.function.name == "run_command":
                command = json.loads(tool_call.function.arguments).get("command", "")
                print(f"[执行命令]: {command}")
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                output = result.stdout.strip() + (
                    "\n" + result.stderr.strip() if result.stderr else ""
                )
            elif tool_call.function.name == "update_todos":
                args = json.loads(tool_call.function.arguments)
                output = update_todos(args["todos"])
            elif tool_call.function.name == "web_fetch":
                args = json.loads(tool_call.function.arguments)
                output = web_fetch(
                    args["url"],
                    args.get("extract_mode", "text"),
                    args.get("max_chars", 8000),
                )
            elif tool_call.function.name == "load_skill":
                args = json.loads(tool_call.function.arguments)
                output = SKILL_LOADER.get_content(args["skill_name"])
            else:
                output = f"未知工具: {tool_call.function.name}"

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
