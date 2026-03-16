from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Union


class NodeType(Enum):
    """Node types for strongly-typed genetic programming [citation:3][citation:10]"""

    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    FUNCTION = "function"
    TERMINAL = "terminal"


@dataclass
class Node:
    """Genetic programming tree node with strong typing"""

    type: NodeType
    value: Union[str, Callable, float]
    children: List["Node"] = field(default_factory=list)
    arity: int = 0

    def __repr__(self) -> str:
        if self.type == NodeType.TERMINAL:
            return str(self.value)
        elif self.type == NodeType.FUNCTION:
            if self.value.__name__ == "<lambda>":
                func_name = self.value.__name__
            else:
                func_name = self.value.__name__
            args = ", ".join(repr(child) for child in self.children)
            return f"{func_name}({args})"
        return f"Node({self.type}, {self.value})"
