import warnings

# Suppress only UserWarnings coming from Pydantic's _fields.py
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._fields"
)
