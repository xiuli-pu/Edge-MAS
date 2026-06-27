from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from edge_mas.agents.prompts import AGENT_PROMPTS
from edge_mas.agents.registry import AGENTS, AUTO_GEN_NAME_MAP, COORDINATOR


def _safe_agent_names() -> list[str]:
    return [COORDINATOR.safe_name] + [agent.safe_name for agent in AGENTS]


async def run_autogen_collaboration(
    task: str,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    max_messages: int = 8,
) -> list[dict[str, str]]:
    """调用 AutoGen 真实多智能体协作。

    本系统把 AutoGen 作为多智能体协作的必要基础，而不是可选增强。
    AutoGen 内部 agent name 必须使用英文安全名，例如 coordinator_agent、literature_agent。
    前端和日志再把英文安全名映射为中文智能体名称。
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except Exception as exc:
        raise RuntimeError(
            "AutoGen 依赖未安装。请先执行：pip install -r requirements.txt。"
        ) from exc

    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
        },
        timeout=180,
    )

    agents = []
    for safe_name in _safe_agent_names():
        display_name = AUTO_GEN_NAME_MAP[safe_name]
        agents.append(
            AssistantAgent(
                name=safe_name,
                model_client=model_client,
                system_message=AGENT_PROMPTS[display_name],
            )
        )

    team = RoundRobinGroupChat(
        agents,
        termination_condition=MaxMessageTermination(max_messages=max_messages),
    )

    rows: list[dict[str, str]] = []
    try:
        async for event in team.run_stream(task=task):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)

            if not source or not content:
                continue

            source_text = str(source)
            if source_text.lower() in {"user", "system"}:
                continue

            rows.append(
                {
                    "agent_safe_name": source_text,
                    "agent": AUTO_GEN_NAME_MAP.get(source_text, source_text),
                    "content": str(content),
                }
            )
    finally:
        try:
            await model_client.close()
        except Exception:
            pass

    if not rows:
        raise RuntimeError("AutoGen 已启动但没有返回有效智能体消息，请检查模型服务是否正常。")

    return rows


def run_autogen_sync(**kwargs: Any) -> list[dict[str, str]]:
    return asyncio.run(run_autogen_collaboration(**kwargs))


def save_autogen_conversation(rows: list[dict[str, str]], run_dir: Path) -> dict[str, Path]:
    """保存 AutoGen 多智能体对话产物。"""
    json_path = run_dir / "autogen_conversation.json"
    md_path = run_dir / "autogen_conversation.md"

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# AutoGen 多智能体协作记录",
        "",
        "本文件由 AutoGen 多智能体框架生成。系统内部使用英文安全 agent 名，前端和文档显示中文智能体名。",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row.get('agent', '智能体')}",
                "",
                str(row.get("content", "")),
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "md": md_path}
