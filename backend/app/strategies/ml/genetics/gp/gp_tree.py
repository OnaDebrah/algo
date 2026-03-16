import logging
import random
from typing import Union

import numpy as np
import pandas as pd

from .function_set import FunctionSet
from .indicators import TechnicalIndicators
from .node_type import Node, NodeType

logger = logging.getLogger(__name__)


class GPTree:
    """
    Genetic Programming Tree representing a trading rule.
    Implements strongly-typed GP to ensure type safety [citation:3][citation:10].
    """

    def __init__(self, max_depth: int = 5, method: str = "grow"):
        """
        Initialize GP tree.

        Args:
            max_depth: Maximum tree depth
            method: 'grow', 'full', or 'ramped_half'
        """
        self.max_depth = max_depth
        self.root = None
        self.fitness = -np.inf
        self.metadata = {}

        self._init_function_set()
        self._init_terminal_set()

        if method != "ramped_half":
            self.root = self._generate_tree(method, depth=0)

    def _init_function_set(self):
        """Initialize function set with strong typing"""
        self.functions = {
            # Arithmetic functions (numeric -> numeric)
            "add": (FunctionSet.add, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "sub": (FunctionSet.sub, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "mul": (FunctionSet.mul, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "div": (FunctionSet.div, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "max": (FunctionSet.max, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "min": (FunctionSet.min, 2, NodeType.NUMERIC, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "abs": (FunctionSet.abs, 1, NodeType.NUMERIC, [NodeType.NUMERIC]),
            "sqrt": (FunctionSet.sqrt, 1, NodeType.NUMERIC, [NodeType.NUMERIC]),
            "log": (FunctionSet.log, 1, NodeType.NUMERIC, [NodeType.NUMERIC]),
            # Relational functions (numeric,numeric -> boolean)
            "gt": (FunctionSet.gt, 2, NodeType.BOOLEAN, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "lt": (FunctionSet.lt, 2, NodeType.BOOLEAN, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "gte": (FunctionSet.gte, 2, NodeType.BOOLEAN, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "lte": (FunctionSet.lte, 2, NodeType.BOOLEAN, [NodeType.NUMERIC, NodeType.NUMERIC]),
            "eq": (FunctionSet.eq, 2, NodeType.BOOLEAN, [NodeType.NUMERIC, NodeType.NUMERIC]),
            # Boolean functions (boolean,boolean -> boolean)
            "And": (FunctionSet.And, 2, NodeType.BOOLEAN, [NodeType.BOOLEAN, NodeType.BOOLEAN]),
            "Or": (FunctionSet.Or, 2, NodeType.BOOLEAN, [NodeType.BOOLEAN, NodeType.BOOLEAN]),
            "Not": (FunctionSet.Not, 1, NodeType.BOOLEAN, [NodeType.BOOLEAN]),
            # Conditional (boolean,any,any -> any) - type determined by true_val/false_val
            "IfThenElse": (FunctionSet.IfThenElse, 3, NodeType.NUMERIC, [NodeType.BOOLEAN, NodeType.NUMERIC, NodeType.NUMERIC]),
            # Lag/difference operators
            "lag": (FunctionSet.lag, 1, NodeType.NUMERIC, [NodeType.NUMERIC]),
            "diff": (FunctionSet.diff, 1, NodeType.NUMERIC, [NodeType.NUMERIC]),
        }

    def _init_terminal_set(self):
        """Initialize terminal set (features and constants)"""
        self.terminals = {
            # Technical indicators (will be computed from data)
            "SMA_10": (lambda data: TechnicalIndicators.SMA(data["Close"], 10), NodeType.NUMERIC),
            "SMA_20": (lambda data: TechnicalIndicators.SMA(data["Close"], 20), NodeType.NUMERIC),
            "SMA_50": (lambda data: TechnicalIndicators.SMA(data["Close"], 50), NodeType.NUMERIC),
            "EMA_10": (lambda data: TechnicalIndicators.EMA(data["Close"], 10), NodeType.NUMERIC),
            "EMA_20": (lambda data: TechnicalIndicators.EMA(data["Close"], 20), NodeType.NUMERIC),
            "EMA_50": (lambda data: TechnicalIndicators.EMA(data["Close"], 50), NodeType.NUMERIC),
            "RSI_14": (lambda data: TechnicalIndicators.RSI(data["Close"], 14), NodeType.NUMERIC),
            "RSI_7": (lambda data: TechnicalIndicators.RSI(data["Close"], 7), NodeType.NUMERIC),
            "MACD_12_26": (lambda data: TechnicalIndicators.MACD(data["Close"], 12, 26), NodeType.NUMERIC),
            "BB_position": (lambda data: TechnicalIndicators.BB_position(data["Close"], 20), NodeType.NUMERIC),
            "ATR_14": (lambda data: TechnicalIndicators.ATR(data["High"], data["Low"], data["Close"], 14), NodeType.NUMERIC),
            "volume_ratio": (lambda data: TechnicalIndicators.volume_ratio(data["Volume"], 20), NodeType.NUMERIC),
            "price_position": (lambda data: TechnicalIndicators.price_position(data["Close"], 20), NodeType.NUMERIC),
            "volatility_20": (lambda data: TechnicalIndicators.volatility(data["Close"], 20), NodeType.NUMERIC),
            "volatility_50": (lambda data: TechnicalIndicators.volatility(data["Close"], 50), NodeType.NUMERIC),
            "DC_001": (lambda data: TechnicalIndicators.directional_change(data["Close"], 0.01), NodeType.NUMERIC),
            "DC_002": (lambda data: TechnicalIndicators.directional_change(data["Close"], 0.02), NodeType.NUMERIC),
            "DC_005": (lambda data: TechnicalIndicators.directional_change(data["Close"], 0.05), NodeType.NUMERIC),
            # Basic price features
            "Close": (lambda data: data["Close"] / data["Close"].iloc[0], NodeType.NUMERIC),  # Normalized
            "Open": (lambda data: data["Open"] / data["Close"].iloc[0], NodeType.NUMERIC),
            "High": (lambda data: data["High"] / data["Close"].iloc[0], NodeType.NUMERIC),
            "Low": (lambda data: data["Low"] / data["Close"].iloc[0], NodeType.NUMERIC),
            "Volume": (lambda data: data["Volume"] / data["Volume"].mean(), NodeType.NUMERIC),
            "returns": (lambda data: data["Close"].pct_change(), NodeType.NUMERIC),
            # Constants
            "0": (0.0, NodeType.NUMERIC),
            "0.5": (0.5, NodeType.NUMERIC),
            "1": (1.0, NodeType.NUMERIC),
            "2": (2.0, NodeType.NUMERIC),
            "5": (5.0, NodeType.NUMERIC),
            "10": (10.0, NodeType.NUMERIC),
            "100": (100.0, NodeType.NUMERIC),
            "True": (True, NodeType.BOOLEAN),
            "False": (False, NodeType.BOOLEAN),
        }

    def _generate_tree(self, method: str, depth: int, expected_type: NodeType = NodeType.BOOLEAN) -> Node:
        """
        Generate tree using specified method.

        Args:
            method: 'grow', 'full', or 'ramped_half'
            depth: Current depth
            expected_type: Expected return type of this subtree

        Returns:
            Root node of generated subtree
        """
        if depth >= self.max_depth - 1:
            # Terminal node
            return self._generate_terminal(expected_type)

        if method == "grow":
            # Mix of functions and terminals
            if random.random() < 0.3:  # 30% chance of terminal
                return self._generate_terminal(expected_type)
            else:
                return self._generate_function(depth, method, expected_type)
        elif method == "full":
            # Full tree - all nodes are functions until max depth
            return self._generate_function(depth, method, expected_type)
        else:
            # Ramped half - will be handled by caller
            return self._generate_function(depth, method, expected_type)

    def _generate_terminal(self, expected_type: NodeType) -> Node:
        """Generate terminal node of specified type"""
        # Filter terminals by type
        candidates = [(name, func_type) for name, (_, func_type) in self.terminals.items() if func_type == expected_type]

        if not candidates:
            # If no matching type, try to find numeric terminal and convert if needed
            if expected_type == NodeType.BOOLEAN:
                # Use comparison with constant
                name, (func, _) = random.choice([(n, f) for n, (f, t) in self.terminals.items() if t == NodeType.NUMERIC])
                # Create comparison node
                left = Node(NodeType.TERMINAL, func, arity=0)
                right = self._generate_terminal(NodeType.NUMERIC)
                return Node(NodeType.FUNCTION, FunctionSet.gt, children=[left, right], arity=2)
            else:
                # Default to numeric terminal
                candidates = [(name, NodeType.NUMERIC) for name, (_, func_type) in self.terminals.items() if func_type == NodeType.NUMERIC]

        name, _ = random.choice(candidates)
        func, _ = self.terminals[name]

        # Handle constant values differently
        if name.replace(".", "").replace("-", "").isdigit() or name in ["True", "False"]:
            value = func if not callable(func) else func
            return Node(NodeType.TERMINAL, value, arity=0)
        else:
            return Node(NodeType.TERMINAL, func, arity=0)

    def _generate_function(self, depth: int, method: str, expected_type: NodeType) -> Node:
        """Generate function node"""
        # Filter functions by return type
        candidates = [
            (name, func, arity, arg_types) for name, (func, arity, ret_type, arg_types) in self.functions.items() if ret_type == expected_type
        ]

        if not candidates:
            # If no matching function, generate terminal
            return self._generate_terminal(expected_type)

        name, func, arity, arg_types = random.choice(candidates)
        children = []
        for i in range(arity):
            child = self._generate_tree(method, depth + 1, arg_types[i])
            children.append(child)

        return Node(NodeType.FUNCTION, func, children=children, arity=arity)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate the GP tree on data.

        Returns:
            Series of trading signals: 1 (buy), -1 (sell), 0 (hold)
        """
        # Add validation
        if data is None or data.empty:
            logger.warning("Empty data passed to GPTree.evaluate")
            return pd.Series(0, index=pd.Index([]))

        if self.root is None:
            return pd.Series(0, index=data.index)

        try:
            # Evaluate tree
            result = self._evaluate_node(self.root, data)

            # Convert to trading signal
            if isinstance(result, pd.Series):
                # Boolean result: True = buy (1), False = sell (-1)
                if result.dtype == bool:
                    return result.astype(int) * 2 - 1  # True->1, False->-1
                # Numeric result: positive = buy, negative = sell, zero = hold
                else:
                    signal = pd.Series(0, index=result.index)
                    signal[result > 0] = 1
                    signal[result < 0] = -1
                    return signal
            else:
                # Scalar result
                if isinstance(result, bool):
                    return pd.Series(1 if result else -1, index=data.index)
                else:
                    if result > 0:
                        return pd.Series(1, index=data.index)
                    elif result < 0:
                        return pd.Series(-1, index=data.index)
                    else:
                        return pd.Series(0, index=data.index)
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error evaluating tree: {e}")
            # Return neutral signal on error
            return pd.Series(0, index=data.index)

    def _evaluate_node(self, node: Node, data: pd.DataFrame) -> Union[float, bool, pd.Series]:
        """Recursively evaluate node"""
        if node.type == NodeType.TERMINAL:
            if callable(node.value):
                return node.value(data)
            else:
                return node.value

        elif node.type == NodeType.FUNCTION:
            # Evaluate children
            args = [self._evaluate_node(child, data) for child in node.children]
            # Apply function
            return node.value(*args)

        else:
            raise ValueError(f"Unknown node type: {node.type}")

    def copy(self) -> "GPTree":
        """Create deep copy of tree"""
        new_tree = GPTree(max_depth=self.max_depth)
        if self.root:
            new_tree.root = self._copy_node(self.root)
        new_tree.fitness = self.fitness
        new_tree.metadata = self.metadata.copy()
        return new_tree

    def _copy_node(self, node: Node) -> Node:
        """Recursively copy node"""
        new_node = Node(node.type, node.value, arity=node.arity)
        for child in node.children:
            new_node.children.append(self._copy_node(child))
        return new_node

    def depth(self) -> int:
        """Calculate tree depth"""
        if self.root is None:
            return 0
        return self._node_depth(self.root)

    def _node_depth(self, node: Node) -> int:
        """Recursive depth calculation"""
        if not node.children:
            return 1
        return 1 + max(self._node_depth(child) for child in node.children)

    def size(self) -> int:
        """Calculate tree size (number of nodes)"""
        if self.root is None:
            return 0
        return self._node_size(self.root)

    def _node_size(self, node: Node) -> int:
        """Recursive size calculation"""
        count = 1
        for child in node.children:
            count += self._node_size(child)
        return count

    def __repr__(self) -> str:
        if self.root is None:
            return "Empty Tree"
        return repr(self.root)
