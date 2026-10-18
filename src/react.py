import ast
import operator
import re


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate(node):
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_calculate(node.operand))
    raise ValueError("Only numeric arithmetic expressions are allowed.")


def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_calculate(tree))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid calculator expression: {exc}") from exc


def react(llm, query, max_cycles=5, min_tool_calls=0, return_details=False):
    if not 1 <= max_cycles <= 5:
        raise ValueError("ReAct must run between one and five cycles.")
    if not 0 <= min_tool_calls < max_cycles:
        raise ValueError("Minimum tool calls must leave a cycle available for the final answer.")
    trace = []
    tokens = 0
    latency = 0.0
    tool_calls = 0
    prompt = (
        "You may use this local tool: calculator(expression).\n"
        "For each cycle, respond with Thought: ... followed by either "
        "Action: calculator[arithmetic expression] or Final answer: ... .\n"
        "Use one calculation per action, carry observations forward, and do not guess "
        "calculator results. If the problem needs multiple calculations, use a separate "
        "action for each calculation before giving the final answer.\n"
        f"Question: {query}"
    )
    for _ in range(max_cycles):
        output, used_tokens, seconds = llm.generate(prompt)
        tokens += used_tokens
        latency += seconds
        trace.append(output)
        if re.search(r"Final answer:", output, re.IGNORECASE):
            if tool_calls < min_tool_calls:
                prompt += (
                    f"\n{output}\nYou have made {tool_calls} calculator calls; "
                    f"make at least {min_tool_calls} separate calculator calls before "
                    "giving your final answer. Continue with the next Action."
                )
                continue
            result = (output, tokens, latency)
            if return_details:
                return (*result, trace)
            return result
        match = re.search(r"Action:\s*calculator\[(.*?)\]", output, re.IGNORECASE)
        if not match:
            raise ValueError(f"ReAct response contained neither an action nor a final answer: {output}")
        observation = calculator(match.group(1))
        tool_calls += 1
        trace.append(f"Observation: {observation}")
        prompt += f"\n{output}\nObservation: {observation}\nContinue."
    raise RuntimeError(
        f"ReAct did not produce a final answer within {max_cycles} cycles. "
        f"Trace: {trace}"
    )
