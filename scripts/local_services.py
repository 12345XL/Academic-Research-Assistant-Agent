"""Manage project-local PostgreSQL and SeaweedFS; never register login services."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local"
PGDATA = STATE / "pgdata"


def executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        for prefix in ("/opt/homebrew/opt/postgresql@18/bin", "/usr/local/opt/postgresql@18/bin"):
            candidate = Path(prefix) / name
            if candidate.exists():
                return str(candidate)
        raise RuntimeError(f"Missing {name}. On macOS: brew install postgresql@18 seaweedfs")
    return found


def config() -> dict[str, str]:
    STATE.mkdir(exist_ok=True, mode=0o700)
    path = ROOT / ".env"
    if not path.exists():
        password = secrets.token_hex(24)
        access = "research" + secrets.token_hex(8)
        secret = secrets.token_hex(32)
        with path.open("x", encoding="utf-8") as output:
            output.write(f"DATABASE_URL=postgresql://research:{password}@127.0.0.1:54329/research_agent\n"
                         f"S3_ENDPOINT_URL=http://127.0.0.1:8333\nS3_ACCESS_KEY_ID={access}\n"
                         f"S3_SECRET_ACCESS_KEY={secret}\nS3_BUCKET=research-papers\nS3_REGION=us-east-1\n")
        path.chmod(0o600)
    values = dict(dotenv_values(path))
    required = ("DATABASE_URL", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET")
    if any(not values.get(key) for key in required):
        raise RuntimeError("Incomplete .env; use .env.example as a guide")
    database = urlparse(values["DATABASE_URL"])
    endpoint = urlparse(values["S3_ENDPOINT_URL"])
    if database.hostname != "127.0.0.1" or database.port != 54329 or endpoint.hostname != "127.0.0.1" or endpoint.port != 8333:
        raise RuntimeError("This manager only operates local ports 54329/8333; start external services separately")
    return values


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def pg_running() -> bool:
    if not PGDATA.exists():
        return False
    return subprocess.run([executable("pg_ctl"), "-D", str(PGDATA), "status"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def weed_pid() -> int | None:
    path = STATE / "seaweed.pid"
    if not path.exists():
        return None
    pid = int(path.read_text())
    result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], text=True, capture_output=True)
    # Never signal an unrelated process after PID reuse.
    return pid if result.returncode == 0 and str(STATE / "seaweed") in result.stdout and "mini" in result.stdout else None


def up(values: dict[str, str]) -> None:
    database = urlparse(values["DATABASE_URL"])
    if not PGDATA.exists():
        password_file = STATE / "init-password"
        password_file.write_text(database.password or "", encoding="utf-8")
        password_file.chmod(0o600)
        try:
            subprocess.run([executable("initdb"), "-D", str(PGDATA), "-U", database.username,
                            "--auth=scram-sha-256", "--encoding=UTF8", "--locale=C",
                            f"--pwfile={password_file}"], check=True, stdout=subprocess.DEVNULL)
        finally:
            password_file.unlink(missing_ok=True)
    if not pg_running():
        if port_open(54329):
            raise RuntimeError("Port 54329 is already owned by another process")
        subprocess.run([executable("pg_ctl"), "-D", str(PGDATA), "-l", str(STATE / "postgres.log"),
                        "-o", "-p 54329 -h 127.0.0.1 -c unix_socket_directories=''", "-w", "start"], check=True)
    import psycopg
    from psycopg import sql
    admin_url = values["DATABASE_URL"].rsplit("/", 1)[0] + "/postgres"
    name = database.path.lstrip("/")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    if not weed_pid():
        for port in (8333, 9333, 9340, 8888, 19333, 19340, 18888, 18333):
            if port_open(port):
                raise RuntimeError(f"SeaweedFS port {port} is occupied; existing service was not changed")
        (STATE / "seaweed").mkdir(exist_ok=True)
        environment = os.environ | {"AWS_ACCESS_KEY_ID": values["S3_ACCESS_KEY_ID"],
                                    "AWS_SECRET_ACCESS_KEY": values["S3_SECRET_ACCESS_KEY"]}
        command = [executable("weed"), "mini", f"-dir={STATE / 'seaweed'}", "-ip=127.0.0.1", "-ip.bind=127.0.0.1",
                   f"-bucket={values['S3_BUCKET']}", "-s3.port=8333", "-webdav=false", "-admin.ui=false",
                   "-s3.port.iceberg=0", "-s3.port.lance=0", "-master.telemetry=false", "-volume.max=8"]
        with (STATE / "seaweed.log").open("ab") as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        (STATE / "seaweed.pid").write_text(str(child.pid))
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url=values["S3_ENDPOINT_URL"],
                          aws_access_key_id=values["S3_ACCESS_KEY_ID"], aws_secret_access_key=values["S3_SECRET_ACCESS_KEY"],
                          region_name="us-east-1", config=Config(connect_timeout=1, read_timeout=1, retries={"max_attempts": 0}))
    for _ in range(40):
        try:
            client.head_bucket(Bucket=values["S3_BUCKET"])
            print("PostgreSQL :54329 and S3 :8333 ready; credentials stay in ignored .env")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("S3 readiness failed; inspect .local/seaweed.log")


def down() -> None:
    pid = weed_pid()
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            if not weed_pid():
                break
            time.sleep(0.25)
    if pg_running():
        subprocess.run([executable("pg_ctl"), "-D", str(PGDATA), "-m", "fast", "-w", "stop"], check=True)
    print("Project services stopped; database and object files retained in .local/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["up", "down", "status"])
    args = parser.parse_args()
    if args.command == "up":
        up(config())
    elif args.command == "down":
        down()
    else:
        print(json.dumps({"postgres_running": pg_running(), "seaweed_running": bool(weed_pid())}))


if __name__ == "__main__":
    main()
