# Allowed imports for strategies
ALLOWED_MODULES = {
    "pandas": ["pd", "DataFrame", "Series"],
    "numpy": ["np", "array", "nan", "inf"],
    "talib": ["*"],  # Technical analysis library
    "datetime": ["datetime", "timedelta"],
    "math": ["*"],
    "statistics": ["mean", "median", "stdev"],
}

# Forbidden operations for security
FORBIDDEN_OPERATIONS = [
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "file",
    "input",
    "raw_input",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "pickle",
    "shelve",
    "marshal",
    "__builtins__",
    "globals",
    "locals",
    "delattr",
    "setattr",
    "getattr",
]
