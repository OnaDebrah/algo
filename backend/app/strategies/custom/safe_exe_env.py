import ast
import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ...strategies.custom.controls import FORBIDDEN_OPERATIONS

logger = logging.getLogger(__name__)


class SafeExecutionEnvironment:
    """Safe environment for executing user code"""

    def __init__(self):
        self.execution_timeout = 30  # seconds
        self.max_memory_mb = 100
        self.FORBIDDEN_OPERATIONS = FORBIDDEN_OPERATIONS

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate user code for security issues

        Returns:
            (is_valid, error_message)
        """
        for forbidden in self.FORBIDDEN_OPERATIONS:
            if forbidden in code:
                return False, f"Forbidden operation detected: {forbidden}"

        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        if "def generate_signals(" not in code:
            return False, "Strategy must define a 'generate_signals(data)' function"

        return True, ""

    def create_safe_globals(self) -> Dict:
        """Create safe global namespace for execution"""
        safe_globals = {
            "__builtins__": {
                "True": True,
                "False": False,
                "None": None,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "print": print,  # Allow print for debugging
            },
            "pd": pd,
            "np": np,
            "datetime": datetime,
        }

        # Add math functions
        import math

        safe_globals["math"] = math

        return safe_globals

    def execute_strategy(self, code: str, data: pd.DataFrame) -> Tuple[bool, Any, str]:
        """
        Execute user strategy code safely

        Returns:
            (success, result, error_message/output)
        """
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return False, None, error

        safe_globals = self.create_safe_globals()
        safe_locals = {}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Execute the strategy code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, safe_locals)

                if "generate_signals" not in safe_locals:
                    return False, None, "Function 'generate_signals' not found in code"

                generate_signals = safe_locals["generate_signals"]

                result = generate_signals(data.copy())

            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            if errors:
                return False, None, f"Execution errors:\n{errors}"

            return True, result, output

        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg
