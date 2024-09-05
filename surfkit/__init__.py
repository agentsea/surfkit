import warnings

# Suppress only UserWarnings coming from Pydantic's _fields.py
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._fields"
)

from surfkit.client import solve
from taskara import Task, TaskStatus
