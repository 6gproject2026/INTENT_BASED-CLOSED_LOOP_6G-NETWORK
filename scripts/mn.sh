#!/usr/bin/env bash
#
# Wrapper around Mininet host/switch commands so they can be run from
# outside the interactive `mininet>` CLI, by shelling into the Docker
# container that hosts the Mininet network and using mnexec to enter a
# host's network namespace.
#
# Usage:
#   ./scripts/mn.sh host G6_D1 ip route replace default via 10.0.0.254
#   ./scripts/mn.sh host G6_D1 ping -c 2 20.0.0.1
#   ./scripts/mn.sh sw ovs-ofctl -O OpenFlow13 dump-ports sp2
#   ./scripts/mn.sh pids
#
# CONTAINER can be overridden via env var, e.g.:
#   CONTAINER=abc123 ./scripts/mn.sh pids

set -euo pipefail

CONTAINER="${CONTAINER:-debc86f1904f}"

usage() {
    cat <<EOF
Usage:
  $0 host <name> <cmd...>   Run <cmd...> inside Mininet host <name>'s namespace
  $0 sw <cmd...>            Run <cmd...> directly in the container (no namespace)
  $0 pids                   List all Mininet host PIDs

Env:
  CONTAINER   Docker container running Mininet (default: debc86f1904f)
EOF
}

# Resolve the single PID for a Mininet host, erroring out clearly if the
# host isn't found or if the pattern is ambiguous (should never happen
# with the trailing '$' anchor, but we guard for it anyway).
resolve_pid() {
    local name="$1"
    local pids
    pids="$(docker exec "$CONTAINER" pgrep -f "mininet:${name}\$" || true)"

    if [ -z "$pids" ]; then
        echo "Error: no Mininet host process found for '${name}' (pattern: mininet:${name}\$)" >&2
        exit 1
    fi

    local count
    count="$(echo "$pids" | wc -l)"
    if [ "$count" -gt 1 ]; then
        echo "Error: pattern 'mininet:${name}\$' matched ${count} PIDs, expected exactly 1:" >&2
        echo "$pids" >&2
        exit 1
    fi

    echo "$pids"
}

cmd_host() {
    if [ "$#" -lt 2 ]; then
        echo "Error: 'host' requires a host name and a command" >&2
        usage >&2
        exit 1
    fi
    local name="$1"
    shift

    local pid
    pid="$(resolve_pid "$name")"

    docker exec "$CONTAINER" mnexec -a "$pid" "$@"
}

cmd_sw() {
    if [ "$#" -lt 1 ]; then
        echo "Error: 'sw' requires a command" >&2
        usage >&2
        exit 1
    fi
    docker exec "$CONTAINER" "$@"
}

cmd_pids() {
    local raw
    raw="$(docker exec "$CONTAINER" pgrep -af 'mininet:' || true)"

    if [ -z "$raw" ]; then
        echo "No Mininet host processes found in container ${CONTAINER}" >&2
        exit 1
    fi

    printf "%-16s %s\n" "HOST" "PID"
    # Each line looks like: "<pid> mininet:<name>"
    echo "$raw" | while IFS= read -r line; do
        local pid host
        pid="${line%% *}"
        host="${line#* mininet:}"
        printf "%-16s %s\n" "$host" "$pid"
    done
}

main() {
    if [ "$#" -lt 1 ]; then
        usage >&2
        exit 1
    fi

    local subcmd="$1"
    shift

    case "$subcmd" in
        host) cmd_host "$@" ;;
        sw) cmd_sw "$@" ;;
        pids) cmd_pids ;;
        -h|--help) usage ;;
        *)
            echo "Error: unknown subcommand '${subcmd}'" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
