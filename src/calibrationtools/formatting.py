from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def get_console() -> Console:
    return Console(force_terminal=True)

def _format_time(seconds: float) -> str:
    """Format time duration in human-readable units.
      - < 1s -> (e.g, 0.5s)
      - < 60s -> integer seconds (e.g. 13s)
      - < 3600s -> MmSs (e.g. 1m10s, 5m00s)
      - < 86400s -> HhMmSs
      - ≥ 86400s -> DdHhMmSs (omit trailing zero units)

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Formatted string with time units.
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    total = int(round(seconds))  # round to nearest whole second
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    if d > 0:
        parts = [f"{d}d"]
        if h > 0:
            parts.append(f"{h}h")
        if m > 0:
            parts.append(f"{m}m")
        if s > 0:
            parts.append(f"{s}s")
        return "".join(parts)
    if h > 0:
        parts = [f"{h}h"]
        if m > 0:
            parts.append(f"{m}m")
        if s > 0:
            parts.append(f"{s}s")
        return "".join(parts)
    if m > 0:
        parts = [f"{m}m"]
        if s > 0:
            parts.append(f"{s}s")
        return "".join(parts)
    return f"{s}s"
