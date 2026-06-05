import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import json

# 加载 .env 文件
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

SYSTEM_PROMPT = """
你是一位充满幽默感的中文 AI 助手，性格开朗、风趣但不失礼貌。  
你的口吻像一个懂科技又喜欢调侃的小伙伴，喜欢用轻松幽默的方式回答问题。  
规则如下：  
1. 回答用户问题时，要保持幽默感，偶尔可以用比喻或小故事，但确保信息准确。  
2. 对于技术问题，你可以用轻松、有趣的语言解释复杂概念，让用户感觉像在听朋友讲故事。  
3. 对于生活、文化、娱乐问题，可以加入风趣评论，但不要冒犯。  
4. 你喜欢用 emoji 增添语气，但不要滥用，保持自然。  
5. 当用户问到你自己的时候，可以自我调侃，但不要透露敏感信息或真实身份。  
示例：
- 用户：“AI 是怎么工作的？”
- AI：“想象一下 AI 是一群勤劳的小精灵，每个精灵都负责一部分计算，它们手拉手把信息从问题搬到答案，最后呈现给你~ 😎”
- 用户：“你是谁？”
- AI：“我？我就是你数字世界的搞笑小伙伴，专门帮你解决问题，又能顺便讲段子 😏”
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "在终端执行一条命令并返回输出",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令"
                }
            },
            "required": ["command"]
        }
    }
}]

def run_command(command:str)->str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr
history = [{"role": "system", "content": SYSTEM_PROMPT}];
while True:
    user_input = input("你:")
    history.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=history,
            max_tokens=2048,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            reply = message.content
            print(f"[agent 回答]: {reply}\n")
            history.append({"role": "assistant", "content": reply})
            break

        history.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            if tool_call.function.name == "run_command":
                args = json.loads(tool_call.function.arguments)
                output = run_command(args["command"])
            else:
                output = f"未知工具: {tool_call.function.name}"

            history.append({"role": "tool","tool_call_id": tool_call.id,"content": output,})
