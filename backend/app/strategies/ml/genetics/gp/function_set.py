from typing import Union

import numpy as np
import pandas as pd


class FunctionSet:
    """
    Function set for genetic programming.
    Defines available operations with strong typing [citation:3][citation:10].
    """

    @staticmethod
    def add(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return a + b

    @staticmethod
    def sub(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return a - b

    @staticmethod
    def mul(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return a * b

    @staticmethod
    def div(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """Protected division to avoid division by zero"""
        if isinstance(b, pd.Series):
            b = b.replace(0, np.nan)
            result = a / b
            return result.fillna(1.0)
        else:
            return a / b if b != 0 else 1.0

    @staticmethod
    def gt(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[bool, pd.Series]:
        """Greater than - returns boolean"""
        return a > b

    @staticmethod
    def lt(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[bool, pd.Series]:
        """Less than - returns boolean"""
        return a < b

    @staticmethod
    def gte(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[bool, pd.Series]:
        """Greater than or equal"""
        return a >= b

    @staticmethod
    def lte(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[bool, pd.Series]:
        """Less than or equal"""
        return a <= b

    @staticmethod
    def eq(a: Union[float, pd.Series], b: Union[float, pd.Series], eps: float = 1e-6) -> Union[bool, pd.Series]:
        """Approximate equality"""
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            return (a - b).abs() < eps
        return abs(a - b) < eps

    @staticmethod
    def And(a: Union[bool, pd.Series], b: Union[bool, pd.Series]) -> Union[bool, pd.Series]:
        """Logical AND - requires boolean inputs"""
        return a & b if isinstance(a, pd.Series) or isinstance(b, pd.Series) else a and b

    @staticmethod
    def Or(a: Union[bool, pd.Series], b: Union[bool, pd.Series]) -> Union[bool, pd.Series]:
        """Logical OR - requires boolean inputs"""
        return a | b if isinstance(a, pd.Series) or isinstance(b, pd.Series) else a or b

    @staticmethod
    def Not(a: Union[bool, pd.Series]) -> Union[bool, pd.Series]:
        """Logical NOT - requires boolean input"""
        return ~a if isinstance(a, pd.Series) else not a

    @staticmethod
    def IfThenElse(cond: Union[bool, pd.Series], true_val: Union[float, pd.Series], false_val: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """If-then-else conditional"""
        if isinstance(cond, pd.Series):
            result = np.where(cond, true_val, false_val)
            return pd.Series(result, index=cond.index)
        return true_val if cond else false_val

    @staticmethod
    def max(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return np.maximum(a, b)

    @staticmethod
    def min(a: Union[float, pd.Series], b: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return np.minimum(a, b)

    @staticmethod
    def abs(a: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return np.abs(a)

    @staticmethod
    def sqrt(a: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """Protected square root"""
        if isinstance(a, pd.Series):
            return a.clip(lower=0).pow(0.5)
        return float(np.sqrt(max(0, int(a))))

    @staticmethod
    def log(a: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """Protected log"""
        if isinstance(a, pd.Series):
            clipped = a.clip(lower=1e-10)
            return pd.Series(np.log(clipped), index=a.index)
        return float(np.log(max(1e-10, a)))

    @staticmethod
    def lag(data: pd.Series, periods: int = 1) -> pd.Series:
        """Lag operator [citation:8]"""
        return data.shift(periods)

    @staticmethod
    def diff(data: Union[float, pd.Series], periods: int = 1) -> Union[float, pd.Series]:
        """Calculate the difference between values"""
        if isinstance(data, pd.Series):
            return data.diff(periods).fillna(0)
        return 0.0
