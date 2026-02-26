#!/usr/bin/env bash
#
# Start all services for the LangGraph RAG project:
#   1. llama.cpp chat model   (port 8080)
#   2. llama.cpp embedding model (port 8081)
#   3. Streamlit web demo     (port 8502)
#
# Usage:
#   ./start.sh          # start all services
#   ./start.sh stop     # stop all services
#   ./start.sh status   # show running services
#   ./start.sh logs     # tail all log files

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/llama-server}"

CHAT_MODEL="${CHAT_MODEL:-$HOME/models/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
EMBED_MODEL="${EMBED_MODEL:-$HOME/models/bge-m3-GGUF/bge-m3-Q8_0.gguf}"

CHAT_PORT="${CHAT_PORT:-8080}"
EMBED_PORT="${EMBED_PORT:-8081}"
DEMO_PORT="${DEMO_PORT:-8502}"

LOG_DIR="$HOME"
CHAT_LOG="$LOG_DIR/llama-chat.log"
EMBED_LOG="$LOG_DIR/llama-embed.log"
DEMO_LOG="$LOG_DIR/streamlit.log"
ES_LOG="$LOG_DIR/elasticsearch.log"

# PID files for clean shutdown
PID_DIR="$SCRIPT_DIR/.pids"

# ── Colours ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

is_port_in_use() {
    ss -tlnp 2>/dev/null | grep -q ":${1} " && return 0 || return 1
}

wait_for_port() {
    local port=$1 name=$2 timeout=${3:-30}
    info "Waiting for $name on port $port ..."
    for (( i=0; i<timeout; i++ )); do
        if is_port_in_use "$port"; then
            info "$name is ready (port $port)"
            return 0
        fi
        sleep 1
    done
    error "$name failed to start within ${timeout}s – check $3"
    return 1
}

save_pid() {
    mkdir -p "$PID_DIR"
    echo "$1" > "$PID_DIR/$2.pid"
}

read_pid() {
    local f="$PID_DIR/$1.pid"
    [[ -f "$f" ]] && cat "$f" || echo ""
}

kill_by_pid() {
    local pid=$1 name=$2
    info "Stopping $name (PID $pid) ..."
    kill "$pid" 2>/dev/null
    for (( i=0; i<5; i++ )); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "Force-killing $name (PID $pid)"
        kill -9 "$pid" 2>/dev/null || true
    fi
}

kill_by_port() {
    local port=$1 name=$2
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [[ -z "$pids" ]]; then
        return 1
    fi
    for pid in $pids; do
        kill_by_pid "$pid" "$name"
    done
    return 0
}

kill_service() {
    local name=$1
    local port=${2:-}
    local pid
    pid=$(read_pid "$name")

    # 1. Try saved PID first
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill_by_pid "$pid" "$name"
        rm -f "$PID_DIR/$name.pid"
        info "$name stopped"
        return
    fi

    rm -f "$PID_DIR/$name.pid"

    # 2. Fall back to port-based kill
    if [[ -n "$port" ]] && kill_by_port "$port" "$name"; then
        info "$name stopped (found via port $port)"
        return
    fi

    warn "$name is not running (no PID or process on port)"
}

# ── Commands ──────────────────────────────────────────────────────────────────

do_stop() {
    info "Stopping all services ..."
    kill_service "streamlit" "$DEMO_PORT"
    kill_service "llama-embed" "$EMBED_PORT"
    kill_service "llama-chat" "$CHAT_PORT"
    if command -v docker &>/dev/null; then
        info "Stopping Elasticsearch ..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" down >> "$ES_LOG" 2>&1 || true
    fi
    info "All services stopped."
}

do_status() {
    if command -v docker &>/dev/null && docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps elasticsearch --status running -q 2>/dev/null | grep -q .; then
        info "elasticsearch  ─  running"
    else
        warn "elasticsearch  ─  not running"
    fi
    declare -A svc_ports=([llama-chat]="$CHAT_PORT" [llama-embed]="$EMBED_PORT" [streamlit]="$DEMO_PORT")
    for svc in llama-chat llama-embed streamlit; do
        local pid port
        pid=$(read_pid "$svc")
        port=${svc_ports[$svc]}
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            info "$svc  ─  running (PID $pid)"
        elif is_port_in_use "$port"; then
            info "$svc  ─  running (detected on port $port)"
        else
            warn "$svc  ─  not running"
        fi
    done
}

do_logs() {
    tail -f "$CHAT_LOG" "$EMBED_LOG" "$DEMO_LOG" 2>/dev/null
}

do_start() {
    # ── Preflight checks ─────────────────────────────────────────────────────

    if [[ ! -x "$LLAMA_SERVER" ]]; then
        error "llama-server not found at $LLAMA_SERVER"
        error "Set LLAMA_SERVER env var or build llama.cpp first."
        exit 1
    fi

    if [[ ! -f "$CHAT_MODEL" ]]; then
        error "Chat model not found: $CHAT_MODEL"
        error "Download it or set CHAT_MODEL env var."
        exit 1
    fi

    if [[ ! -f "$EMBED_MODEL" ]]; then
        error "Embedding model not found: $EMBED_MODEL"
        error "Download it or set EMBED_MODEL env var."
        exit 1
    fi

    if ! command -v uv &>/dev/null; then
        error "'uv' is not installed. Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # ── 0. Elasticsearch ────────────────────────────────────────────────────

    if ! command -v docker &>/dev/null; then
        warn "docker not found – skipping Elasticsearch (install Docker or start ES manually)"
    elif docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps elasticsearch --status running -q 2>/dev/null | grep -q .; then
        warn "Elasticsearch already running – skipping"
    else
        info "Starting Elasticsearch via Docker Compose ..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d elasticsearch >> "$ES_LOG" 2>&1
        wait_for_port 9200 "Elasticsearch" 30
    fi

    # ── 1. Chat model server ─────────────────────────────────────────────────

    if is_port_in_use "$CHAT_PORT"; then
        warn "Port $CHAT_PORT already in use – skipping chat model server"
    else
        info "Starting chat model server on port $CHAT_PORT ..."
        nohup "$LLAMA_SERVER" \
            --model "$CHAT_MODEL" \
            --ctx-size 16384 \
            --host 0.0.0.0 --port "$CHAT_PORT" \
            --seed 3407 --temp 0.7 --top-p 1.0 --min-p 0.01 \
            --jinja --reasoning-budget 0 \
            > "$CHAT_LOG" 2>&1 &
        save_pid $! "llama-chat"
        wait_for_port "$CHAT_PORT" "Chat model" 60
    fi

    # ── 2. Embedding model server ────────────────────────────────────────────

    if is_port_in_use "$EMBED_PORT"; then
        warn "Port $EMBED_PORT already in use – skipping embedding model server"
    else
        info "Starting embedding model server on port $EMBED_PORT ..."
        nohup "$LLAMA_SERVER" \
            --model "$EMBED_MODEL" \
            --ctx-size 8192 \
            --host 0.0.0.0 --port "$EMBED_PORT" \
            --embedding \
            > "$EMBED_LOG" 2>&1 &
        save_pid $! "llama-embed"
        wait_for_port "$EMBED_PORT" "Embedding model" 60
    fi

    # ── 3. Streamlit web demo ────────────────────────────────────────────────

    if is_port_in_use "$DEMO_PORT"; then
        warn "Port $DEMO_PORT already in use – skipping Streamlit demo"
    else
        info "Starting Streamlit web demo on port $DEMO_PORT ..."
        cd "$SCRIPT_DIR"
        nohup uv run streamlit run demo.py \
            --server.address 0.0.0.0 \
            --server.port "$DEMO_PORT" \
            > "$DEMO_LOG" 2>&1 &
        save_pid $! "streamlit"
        wait_for_port "$DEMO_PORT" "Streamlit demo" 30
    fi

    # ── Summary ──────────────────────────────────────────────────────────────

    echo ""
    info "All services started!"
    echo "  ┌──────────────────────────────────────────────────┐"
    echo "  │  Elasticsearch       http://localhost:9200       │"
    echo "  │  Chat model server   http://localhost:$CHAT_PORT      │"
    echo "  │  Embedding server    http://localhost:$EMBED_PORT      │"
    echo "  │  Web demo            http://localhost:$DEMO_PORT      │"
    echo "  └──────────────────────────────────────────────────┘"
    echo ""
    info "Logs: $CHAT_LOG, $EMBED_LOG, $DEMO_LOG"
    info "Stop all: $0 stop"
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "${1:-start}" in
    start)  do_start  ;;
    stop)   do_stop   ;;
    status) do_status ;;
    logs)   do_logs   ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        exit 1
        ;;
esac
