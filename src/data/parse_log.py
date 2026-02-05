"""
Parse NASA log files from raw text to parsed CSV
Handles malformed log entries with robust parsing
"""

import re
import pandas as pd
from pathlib import Path
from typing import Optional, Dict


# Regex patterns
STRICT_PATTERN = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)$'
)

BASIC_HEAD = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+(?P<rest>.*)$'
)

STATUS_BYTES_TAIL = re.compile(r'(?P<status>\d{3})\s+(?P<bytes>\S+)\s*$')
METHOD_RE = re.compile(r'\b(?P<method>GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH)\b')
HTTPVER_RE = re.compile(r'HTTP/\d\.\d')


def _to_bytes(x: str) -> int:
    """Convert bytes string to int, treating '-' as 0"""
    return 0 if x == "-" else int(x)


def salvage_request(rest: str) -> Optional[str]:
    """
    Salvage malformed request strings like:
    - '"GET /path/" HTTP/1.0" 404 -'
    - '"GET /images/">index of /images HTTP/1.0" 404 -'
    
    Returns cleaned request: 'METHOD URL HTTP/1.0' or None
    """
    # Remove status+bytes from end
    sb = STATUS_BYTES_TAIL.search(rest)
    rest_wo_tail = rest[:sb.start()].strip() if sb else rest.strip()

    # Find method
    m_method = METHOD_RE.search(rest_wo_tail)
    if not m_method:
        return None
    method = m_method.group("method")

    # Find HTTP version (take the last one)
    http_all = list(HTTPVER_RE.finditer(rest_wo_tail))
    if not http_all:
        return None
    httpver = http_all[-1].group(0)

    # Extract URL between method and httpver
    start = m_method.end()
    end = http_all[-1].start()
    url_raw = rest_wo_tail[start:end]

    # Clean URL
    url = url_raw.replace('"', "").strip()
    for cut in [">", "<"]:
        if cut in url:
            url = url.split(cut, 1)[0].strip()

    if not url or url == "/":
        url = "/" if "/" in url_raw else url.strip()

    if not url:
        return None

    # URL should not have spaces (take first part)
    url = url.split()[0]

    return f"{method} {url} {httpver}"


def parse_log_line_robust(line: str) -> Optional[Dict]:
    """
    Parse log line: try strict pattern first, then salvage malformed lines
    Returns: dict with host, timestamp, request, status, bytes
    """
    line = line.strip()
    if not line:
        return None

    # Try strict pattern first
    m = STRICT_PATTERN.match(line)
    if m:
        d = m.groupdict()
        d["status"] = int(d["status"])
        d["bytes"] = _to_bytes(d["bytes"])
        return d

    # Try to salvage
    h = BASIC_HEAD.match(line)
    if not h:
        return None

    host = h.group("host")
    timestamp = h.group("timestamp")
    rest = h.group("rest")

    sb = STATUS_BYTES_TAIL.search(rest)
    if not sb:
        return None

    status = int(sb.group("status"))
    bytes_ = _to_bytes(sb.group("bytes"))

    request = salvage_request(rest)
    if request is None:
        request = ""

    return {
        "host": host,
        "timestamp": timestamp,
        "request": request,
        "status": status,
        "bytes": bytes_,
    }


def parse_log_file(input_path: str) -> pd.DataFrame:
    """
    Parse NASA log file to DataFrame
    
    Args:
        input_path: Path to raw log file
        
    Returns:
        DataFrame with columns: host, timestamp, request, status, bytes
    """
    rows = []
    bad_lines = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            item = parse_log_line_robust(line)
            if item is None:
                bad_lines += 1
                continue
            rows.append(item)

    df = pd.DataFrame(rows)
    print(f"✓ Parsed {len(df):,} rows | Skipped {bad_lines:,} bad lines")
    
    return df


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add parsed columns: datetime, method, url, version
    
    Args:
        df: DataFrame with raw columns
        
    Returns:
        DataFrame with additional parsed columns
    """
    # Parse timestamp to datetime
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%b/%Y:%H:%M:%S %z",
        errors="coerce"
    )
    
    # Parse request into method, url, version
    request_pattern = re.compile(
        r'^(?P<method>GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH)\s+'
        r'(?P<url>\S+)\s+'
        r'(?P<version>HTTP/\d\.\d)$'
    )
    
    def parse_request(req):
        if pd.isna(req) or req == "":
            return pd.Series(["UNKNOWN", "/", "HTTP/1.0"])
        m = request_pattern.match(req.strip())
        if m:
            return pd.Series([m.group("method"), m.group("url"), m.group("version")])
        return pd.Series(["UNKNOWN", "/", "HTTP/1.0"])
    
    df[["method", "url", "version"]] = df["request"].apply(parse_request)
    
    return df


def parse_and_save(input_path: str, output_path: str):
    """
    Main pipeline: parse log file and save to CSV
    
    Args:
        input_path: Path to raw .txt log file
        output_path: Path to save parsed .csv file
    """
    print(f"\n{'='*60}")
    print(f"Parsing: {input_path}")
    print(f"{'='*60}")
    
    # Parse log file
    df = parse_log_file(input_path)
    
    # Add parsed columns
    df = add_parsed_columns(df)
    
    # Sort by datetime
    df = df.sort_values(by="datetime").reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print()
    
    return df


if __name__ == "__main__":
    # Parse train and test files
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".txt", "_parsed.csv")
        parse_and_save(input_file, output_file)
    else:
        # Default paths
        parse_and_save("data/raw/train.txt", "data/interim/train_parsed.csv")
        parse_and_save("data/raw/test.txt", "data/interim/test_parsed.csv")