import json
import re
import socket
from datetime import datetime
from typing import Any, Dict, Optional


def extract_parse_json(input_str: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from the input string using regex if it is tagged with 'json\n'
    and enclosed in backticks, otherwise returns the input string.

    :param input_str: A string that may contain a JSON object.
    :return: A dictionary if JSON is parsed, otherwise the original string.
    """
    # Regex to match 'json\n{...}' pattern enclosed in backticks
    match = re.search(r"```json\n([\s\S]+?)\n```", input_str)
    if match:
        json_str = match.group(1)  # Extract the JSON string
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise
    else:
        return json.loads(input_str)


def find_open_port(start_port: int = 1024, end_port: int = 65535) -> Optional[int]:
    """Finds an open port on the machine"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port  # Port is open
            except socket.error:
                continue  # Port is in use, try the next one
    return None  # No open port found


def convert_unix_to_datetime(unix_timestamp: int) -> str:
    dt = datetime.utcfromtimestamp(unix_timestamp)
    friendly_format = dt.strftime("%Y-%m-%d %H:%M:%S")
    return friendly_format
