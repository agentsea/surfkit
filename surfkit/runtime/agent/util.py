import random
import string

from namesgenerator import get_random_name

from surfkit.types import AgentType


def instance_name(type: AgentType) -> str:
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    random_name = get_random_name("-")
    if not random_name:
        raise ValueError("Could not generate a random name")
    name_parts = random_name.split("-")
    if len(name_parts) != 2:
        raise ValueError("Could not generate a random name with 2 parts")
    name_only = name_parts[1]

    return f"{type.name.lower()}-{name_only.lower()}-{random_string.lower()}"
