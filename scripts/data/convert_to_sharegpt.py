#!/usr/bin/env python3
"""
Convert Cyber-AutoAgent session data to ShareGPT format for fine-tuning.

Supports two output formats:
1. ShareGPT format (for general fine-tuning)
2. OpenAI tool-calling format (for GLM-4.7-Flash with tool use)
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from various content formats."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Handle toolUse
                if "toolUse" in item:
                    tool = item["toolUse"]
                    tool_name = tool.get("name", "unknown")
                    tool_input = tool.get("input", {})

                    # Format based on tool type
                    if tool_name == "shell":
                        commands = tool_input.get("command", [])
                        if isinstance(commands, list):
                            cmd_str = "\n".join(commands)
                        else:
                            cmd_str = str(commands)
                        parts.append(f"<tool_call>\n{tool_name}: {cmd_str}\n</tool_call>")
                    else:
                        parts.append(f"<tool_call>\n{tool_name}: {json.dumps(tool_input, indent=2)}\n</tool_call>")

                # Handle toolResult
                elif "toolResult" in item:
                    result = item["toolResult"]
                    status = result.get("status", "unknown")
                    result_content = result.get("content", [])

                    text_parts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and "text" in rc:
                            text_parts.append(rc["text"])
                        elif isinstance(rc, str):
                            text_parts.append(rc)

                    parts.append(f"<tool_result status=\"{status}\">\n{chr(10).join(text_parts)}\n</tool_result>")

                # Handle text content
                elif "text" in item:
                    parts.append(item["text"])
                elif "type" in item and item["type"] == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts)

    return str(content)


def extract_tool_calls_openai(content: Any) -> tuple[str, List[Dict]]:
    """Extract tool calls in OpenAI format from content."""
    tool_calls = []
    text_content = ""

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if "toolUse" in item:
                    tool = item["toolUse"]
                    tool_name = tool.get("name", "unknown")
                    tool_input = tool.get("input", {})
                    tool_id = tool.get("toolUseId", f"call_{len(tool_calls)}")

                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input)
                        }
                    })
                elif "text" in item:
                    text_content += item["text"]
    elif isinstance(content, str):
        text_content = content

    return text_content, tool_calls


def extract_tool_result_openai(content: Any) -> tuple[str, str, str]:
    """Extract tool result in OpenAI format. Returns (tool_call_id, tool_name, result_text)."""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "toolResult" in item:
                result = item["toolResult"]
                tool_id = result.get("toolUseId", "")
                result_content = result.get("content", [])

                text_parts = []
                for rc in result_content:
                    if isinstance(rc, dict) and "text" in rc:
                        text_parts.append(rc["text"])
                    elif isinstance(rc, str):
                        text_parts.append(rc)

                return tool_id, "", "\n".join(text_parts)

    return "", "", ""


def load_session(session_dir: Path) -> List[Dict[str, Any]]:
    """Load all messages from a session directory."""
    messages = []

    # Find the agent directory
    agents_dir = session_dir / "agents"
    if not agents_dir.exists():
        return messages

    for agent_dir in agents_dir.iterdir():
        if not agent_dir.is_dir():
            continue

        messages_dir = agent_dir / "messages"
        if not messages_dir.exists():
            continue

        # Load all message files
        message_files = sorted(
            messages_dir.glob("message_*.json"),
            key=lambda x: int(x.stem.split("_")[1])
        )

        for msg_file in message_files:
            try:
                with open(msg_file, "r") as f:
                    data = json.load(f)
                    if "message" in data:
                        messages.append(data["message"])
            except Exception as e:
                print(f"Error loading {msg_file}: {e}")

    return messages


def convert_to_sharegpt(messages: List[Dict], system_prompt: Optional[str] = None) -> Dict:
    """Convert messages to ShareGPT format."""
    conversations = []

    if system_prompt:
        conversations.append({
            "from": "system",
            "value": system_prompt
        })

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        text = extract_text_from_content(content)
        if not text.strip():
            continue

        if role == "assistant":
            conversations.append({"from": "gpt", "value": text})
        elif role == "user":
            conversations.append({"from": "human", "value": text})
        elif role == "system":
            conversations.append({"from": "system", "value": text})

    return {"conversations": conversations}


def convert_to_openai_tools(messages: List[Dict], system_prompt: Optional[str] = None) -> Dict:
    """Convert messages to OpenAI tool-calling format (for GLM-4.7-Flash)."""
    converted = []

    if system_prompt:
        converted.append({
            "role": "system",
            "content": system_prompt
        })

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            text_content, tool_calls = extract_tool_calls_openai(content)

            msg_dict = {"role": "assistant"}
            if text_content:
                msg_dict["content"] = text_content
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls

            if text_content or tool_calls:
                converted.append(msg_dict)

        elif role == "user":
            # Check if this is a tool result
            tool_id, tool_name, result_text = extract_tool_result_openai(content)

            if tool_id:
                # This is a tool result
                converted.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text
                })
            else:
                # Regular user message
                text = extract_text_from_content(content)
                if text.strip():
                    converted.append({
                        "role": "user",
                        "content": text
                    })

        elif role == "system":
            text = extract_text_from_content(content)
            if text.strip():
                converted.append({
                    "role": "system",
                    "content": text
                })

        i += 1

    return {"messages": converted}


def check_for_flag(messages: List[Dict]) -> bool:
    """Check if the session contains a captured flag."""
    flag_patterns = [r"XBOW\{[^}]+\}", r"FLAG\{[^}]+\}", r"flag\{[^}]+\}"]

    for msg in messages:
        content = msg.get("content", "")
        text = extract_text_from_content(content)

        for pattern in flag_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Convert CAA sessions to ShareGPT format")
    parser.add_argument(
        "--input", "-i",
        default="scripts/train/training_data/sessions",
        help="Input directory containing session folders"
    )
    parser.add_argument(
        "--output", "-o",
        default="scripts/train/train_sharegpt.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["sharegpt", "openai_tools"],
        default="openai_tools",
        help="Output format (default: openai_tools for GLM-4.7-Flash)"
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include sessions where a flag was captured"
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a skilled cybersecurity expert participating in a CTF challenge. Your goal is to find and capture the flag by exploiting vulnerabilities in the target application. Use systematic reconnaissance, identify attack vectors, and execute exploits carefully.",
        help="System prompt to prepend"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Process all sessions
    all_data = []
    success_count = 0
    total_count = 0

    for session_dir in sorted(input_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue

        total_count += 1
        print(f"Processing: {session_dir.name}")

        messages = load_session(session_dir)
        if not messages:
            print(f"  -> No messages found, skipping")
            continue

        has_flag = check_for_flag(messages)
        if has_flag:
            success_count += 1
            print(f"  -> Flag found! ({len(messages)} messages)")
        else:
            print(f"  -> No flag ({len(messages)} messages)")
            if args.success_only:
                continue

        # Convert based on format
        if args.format == "sharegpt":
            converted = convert_to_sharegpt(messages, args.system_prompt)
        else:
            converted = convert_to_openai_tools(messages, args.system_prompt)

        # Add metadata
        converted["metadata"] = {
            "session": session_dir.name,
            "success": has_flag,
            "message_count": len(messages)
        }

        all_data.append(converted)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Total sessions: {total_count}")
    print(f"  Successful (flag captured): {success_count}")
    print(f"  Output samples: {len(all_data)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    main()
