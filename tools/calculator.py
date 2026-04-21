"""
Calculator tool: safely evaluates mathematical expressions.

Uses Python's `ast` module to parse expressions and only allow
a whitelist of operations (arithmetic, math functions, etc.).
Never uses eval() directly — that would be a remote code execution
vulnerability in an agent context where expressions come from the LLM.

Supports:
  - Arithmetic: + - * / // % **
  - Unary: -x, +x
  - Math functions: sqrt, log, exp, sin, cos, tan, pi, e, etc.
  - Built-ins: abs, round, min, max, sum, pow
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from tools.base import Tool


# Allowed binary operators
_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed names (math constants + safe functions)
_ALLOWED_NAMES: dict[str, Any] = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    # Functions
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "degrees": math.degrees, "radians": math.radians,
    # Safe built-ins
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow,
}


class CalculatorError(Exception):
    """Raised when an expression is unsafe or malformed."""


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node, raising on anything not whitelisted."""
    # Python literal (number, string, etc.)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise CalculatorError(f"Unsupported constant type: {type(node.value).__name__}")

    # Binary operation: a + b, a * b, etc.
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"Operator {op_type.__name__} not allowed")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _BIN_OPS[op_type](left, right)

    # Unary operation: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"Unary operator {op_type.__name__} not allowed")
        operand = _safe_eval(node.operand)
        return _UNARY_OPS[op_type](operand)

    # Function call: sqrt(9), log(100, 10), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only simple function calls allowed")
        fn_name = node.func.id
        if fn_name not in _ALLOWED_NAMES:
            raise CalculatorError(f"Function '{fn_name}' not allowed")
        fn = _ALLOWED_NAMES[fn_name]
        if not callable(fn):
            raise CalculatorError(f"'{fn_name}' is not callable")
        args = [_safe_eval(a) for a in node.args]
        if node.keywords:
            raise CalculatorError("Keyword arguments not allowed")
        return fn(*args)

    # Named constant: pi, e
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise CalculatorError(f"Name '{node.id}' not allowed")
        val = _ALLOWED_NAMES[node.id]
        if callable(val):
            raise CalculatorError(f"'{node.id}' is a function, not a value")
        return val

    raise CalculatorError(f"Unsupported AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    """Parse and evaluate a mathematical expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Syntax error: {e}")
    return _safe_eval(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluates a mathematical expression and returns the numerical result. "
        "Supports arithmetic (+, -, *, /, //, %, **), math functions "
        "(sqrt, log, exp, sin, cos, tan, floor, ceil, factorial), and "
        "constants (pi, e). Use this for any numeric computation — "
        "do NOT do arithmetic in your head. "
        "Example expressions: '5000 * (1.045 ** 7)', 'sqrt(144)', '2 * pi * 5'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 2 * 3' or 'sqrt(169)'.",
            }
        },
        "required": ["expression"],
    }

    async def _run(self, expression: str) -> str:
        result = safe_calculate(expression)
        # Format nicely: int if integer-valued, else float with reasonable precision
        if isinstance(result, float) and result.is_integer():
            return f"{expression} = {int(result)}"
        if isinstance(result, float):
            return f"{expression} = {result:.10g}"
        return f"{expression} = {result}"