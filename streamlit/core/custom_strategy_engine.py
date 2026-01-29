"""
Custom Strategy Engine - Execute user-defined strategies with safety checks
"""

import ast
import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class SafeExecutionEnvironment:
    """Safe environment for executing user code"""

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

    def __init__(self):
        self.execution_timeout = 30  # seconds
        self.max_memory_mb = 100

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate user code for security issues

        Returns:
            (is_valid, error_message)
        """
        # Check for forbidden operations
        for forbidden in self.FORBIDDEN_OPERATIONS:
            if forbidden in code:
                return False, f"Forbidden operation detected: {forbidden}"

        # Try to parse as valid Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        # Check for required function signature
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
        # Validate code first
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return False, None, error

        # Create safe execution environment
        safe_globals = self.create_safe_globals()
        safe_locals = {}

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Execute the strategy code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, safe_locals)

                # Get the generate_signals function
                if "generate_signals" not in safe_locals:
                    return False, None, "Function 'generate_signals' not found in code"

                generate_signals = safe_locals["generate_signals"]

                # Execute the strategy
                result = generate_signals(data.copy())

            # Get captured output
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            if errors:
                return False, None, f"Execution errors:\n{errors}"

            return True, result, output

        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg


class StrategyCodeGenerator:
    """Generate strategy code from natural language prompts using AI"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)

    async def generate_strategy_code(
        self,
        prompt: str,
        style: str = "technical",  # technical, fundamental, hybrid
    ) -> Tuple[str, str, str]:
        """
        Generate strategy code from natural language

        Returns:
            (code, explanation, example_usage)
        """
        if not self.client:
            return self._generate_template_strategy(prompt, style)

        # Build AI prompt
        ai_prompt = self._build_code_generation_prompt(prompt, style)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.3,
                messages=[{"role": "user", "content": ai_prompt}],
            )

            response = message.content[0].text

            # Parse response
            code, explanation, example = self._parse_ai_response(response)

            return code, explanation, example

        except Exception as e:
            print(f"AI generation error: {e}")
            return self._generate_template_strategy(prompt, style)

    def _build_code_generation_prompt(self, prompt: str, style: str) -> str:
        """Build prompt for Claude to generate strategy code"""

        system_context = f"""You are an expert quantitative trading strategy developer.
        Generate Python code for a trading strategy based on the user's description.

IMPORTANT CONSTRAINTS:
1. The code must define a function: generate_signals(data)
2. Input 'data' is a pandas DataFrame with columns: Open, High, Low, Close, Volume
3. Output must be a pandas Series with values: 1 (buy), -1 (sell), 0 (hold)
4. Only use: pandas (pd), numpy (np), math, datetime
5. NO external API calls, file I/O, or network operations
6. Keep code under 100 lines
7. Include comments explaining the logic
8. Make it efficient and vectorized (avoid loops when possible)

STRATEGY STYLE: {style}

USER REQUEST: {prompt}

Respond with valid Python code that can be executed directly. Structure your response as:

```python
# [Strategy Name]
# [One-line description]

def generate_signals(data):
    \"\"\"
    [Detailed description of the strategy logic]

    Args:
    data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"

    # Your implementation here

    return signals
```

Then after the code, provide:

EXPLANATION:
[2-3 paragraphs explaining the strategy, when it works best, and key parameters]

EXAMPLE USAGE:
[Show how to use the strategy with sample code]

Make the strategy practical, well-documented, and production-ready."""

        return system_context

    def _parse_ai_response(self, response: str) -> Tuple[str, str, str]:
        """Parse AI response into code, explanation, and example"""

        # Extract code block
        code_pattern = r"```python\n(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL)

        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response  # Fallback: assume entire response is code

        # Extract explanation
        explanation_pattern = r"EXPLANATION:\n(.*?)(?:EXAMPLE USAGE:|$)"
        explanation_match = re.search(explanation_pattern, response, re.DOTALL)

        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = "AI-generated trading strategy based on your requirements."

        # Extract example
        example_pattern = r"EXAMPLE USAGE:\n(.*?)$"
        example_match = re.search(example_pattern, response, re.DOTALL)

        if example_match:
            example = example_match.group(1).strip()
        else:
            example = "signals = generate_signals(data)"

        return code, explanation, example

    def _generate_template_strategy(self, prompt: str, style: str) -> Tuple[str, str, str]:
        """Generate template strategy when AI is unavailable"""

        # Detect strategy type from prompt
        prompt_lower = prompt.lower()

        if "sma" in prompt_lower or "moving average" in prompt_lower:
            return self._sma_crossover_template()
        elif "rsi" in prompt_lower:
            return self._rsi_template()
        elif "macd" in prompt_lower:
            return self._macd_template()
        elif "bollinger" in prompt_lower:
            return self._bollinger_template()
        else:
            return self._simple_momentum_template()

    def _sma_crossover_template(self) -> Tuple[str, str, str]:
        """SMA crossover strategy template"""
        code = """# SMA Crossover Strategy
# Buy when fast MA crosses above slow MA, sell when crosses below

def generate_signals(data):
    \"\"\"
    Simple Moving Average crossover strategy.
    Generates buy signals when fast SMA crosses above slow SMA.

    Args:
        data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # Calculate moving averages
    fast_period = 20
    slow_period = 50

    data['SMA_fast'] = data['Close'].rolling(window=fast_period).mean()
    data['SMA_slow'] = data['Close'].rolling(window=slow_period).mean()

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Generate signals
    # Buy when fast crosses above slow
    signals[(data['SMA_fast'] > data['SMA_slow']) &
            (data['SMA_fast'].shift(1) <= data['SMA_slow'].shift(1))] = 1

    # Sell when fast crosses below slow
    signals[(data['SMA_fast'] < data['SMA_slow']) &
            (data['SMA_fast'].shift(1) >= data['SMA_slow'].shift(1))] = -1

    return signals
"""

        explanation = """This is a classic trend-following strategy using Simple Moving Average (SMA) crossovers.

The strategy uses two moving averages: a fast (20-period) and slow (50-period). When the fast MA crosses above the slow MA,
it signals an uptrend and generates a buy signal. When the fast MA crosses below the slow MA, it signals a downtrend
and generates a sell signal.

This strategy works best in trending markets and may produce false signals in choppy, sideways markets.
Consider adding volume or other filters to reduce whipsaws."""

        example = """# Example usage:
import pandas as pd
signals = generate_signals(data)
buy_dates = data.index[signals == 1]
sell_dates = data.index[signals == -1]
print(f"Buy signals: {len(buy_dates)}, Sell signals: {len(sell_dates)}")"""

        return code, explanation, example

    def _rsi_template(self) -> Tuple[str, str, str]:
        """RSI mean reversion strategy template"""
        code = """# RSI Mean Reversion Strategy
# Buy when oversold, sell when overbought

def generate_signals(data):
    \"\"\"
    RSI-based mean reversion strategy.
    Buys when RSI is oversold and sells when overbought.

    Args:
        data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # Calculate RSI
    period = 14
    oversold = 30
    overbought = 70

    # Price changes
    delta = data['Close'].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Generate signals
    # Buy when RSI crosses above oversold level
    signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1

    # Sell when RSI crosses below overbought level
    signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1

    return signals
"""

        explanation = """RSI (Relative Strength Index) mean reversion strategy that exploits overbought/oversold conditions.

The strategy buys when RSI rises above the oversold threshold (30), indicating potential reversal from oversold conditions.
It sells when RSI falls below the overbought threshold (70), indicating potential reversal from overbought conditions.

This strategy works best in ranging markets but can struggle during strong trends. Consider combining with trend filters
or adjusting thresholds based on volatility."""

        example = """# Example usage:
signals = generate_signals(data)
positions = signals.cumsum()  # Track position
returns = (data['Close'].pct_change() * positions.shift()).cumsum()"""

        return code, explanation, example

    def _macd_template(self) -> Tuple[str, str, str]:
        """MACD momentum strategy template"""
        code = """# MACD Momentum Strategy
# Trade based on MACD crossovers and momentum

def generate_signals(data):
    \"\"\"
    MACD momentum strategy using signal line crossovers.

    Args:
        data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # MACD parameters
    fast = 12
    slow = 26
    signal = 9

    # Calculate MACD
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Generate signals
    # Buy when MACD crosses above signal line
    signals[(macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))] = 1

    # Sell when MACD crosses below signal line
    signals[(macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))] = -1

    return signals
"""

        explanation = """MACD (Moving Average Convergence Divergence) momentum strategy based on signal line crossovers.

The strategy generates buy signals when the MACD line crosses above the signal line, indicating bullish momentum.
Sell signals occur when the MACD crosses below the signal line, indicating bearish momentum.

This strategy is effective in trending markets and provides good entry/exit timing.
Can be enhanced with histogram analysis or divergence detection."""

        example = """# Example with histogram analysis:
signals = generate_signals(data)
# Could add: filter signals by MACD histogram strength
# strong_signals = signals[abs(histogram) > threshold]"""

        return code, explanation, example

    def _bollinger_template(self) -> Tuple[str, str, str]:
        """Bollinger Bands strategy template"""
        code = """# Bollinger Bands Mean Reversion
# Trade band touches and reversals

def generate_signals(data):
    \"\"\"
    Bollinger Bands mean reversion strategy.

    Args:
        data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # Bollinger Bands parameters
    period = 20
    std_dev = 2

    # Calculate Bollinger Bands
    middle_band = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Generate signals
    # Buy when price touches lower band and bounces up
    signals[(data['Close'] <= lower_band) &
            (data['Close'].shift(-1) > lower_band)] = 1

    # Sell when price touches upper band and bounces down
    signals[(data['Close'] >= upper_band) &
            (data['Close'].shift(-1) < upper_band)] = -1

    return signals
"""

        explanation = """Bollinger Bands mean reversion strategy that trades extreme price moves.

The strategy buys when price touches or breaks below the lower band and shows signs of reversal, expecting mean reversion.
It sells when price touches or breaks above the upper band and reverses.

Works best in ranging markets with regular oscillations. In strong trends, prices can "walk the bands" causing false signals.
Consider adding trend filters or adjusting standard deviation multiplier."""

        example = """# Example with band width filter:
signals = generate_signals(data)
# Could add: avoid trades when bands too narrow (low volatility)
# band_width = (upper - lower) / middle
# signals = signals[band_width > threshold]"""

        return code, explanation, example

    def _simple_momentum_template(self) -> Tuple[str, str, str]:
        """Simple momentum strategy template"""
        code = """# Simple Momentum Strategy
# Buy strong momentum, sell weak momentum

def generate_signals(data):
    \"\"\"
    Simple momentum strategy based on rate of change.

    Args:
        data: pandas DataFrame with OHLCV columns

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # Calculate momentum (rate of change)
    lookback = 20
    momentum = data['Close'].pct_change(lookback) * 100

    # Momentum thresholds
    buy_threshold = 5.0  # 5% gain over lookback
    sell_threshold = -5.0  # 5% loss over lookback

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Generate signals
    # Buy on strong positive momentum
    signals[(momentum > buy_threshold) &
            (momentum.shift(1) <= buy_threshold)] = 1

    # Sell on strong negative momentum
    signals[(momentum < sell_threshold) &
            (momentum.shift(1) >= sell_threshold)] = -1

    return signals
"""

        explanation = """Simple momentum strategy that trades on strong price movements.

The strategy buys when the security shows strong upward momentum (price gain over lookback period exceeds threshold)
and sells when it shows strong downward momentum.

This is a trend-following approach that works well in trending markets but can suffer during consolidations.
Can be improved by adding volume confirmation or multiple timeframe analysis."""

        example = """# Example with volume filter:
signals = generate_signals(data)
# Could add: require high volume on signal days
# volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
# signals = signals[volume_ratio > 1.5]"""

        return code, explanation, example
