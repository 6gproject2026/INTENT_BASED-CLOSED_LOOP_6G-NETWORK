# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (use the venv in ./venv)
pip install -r requirements.txt

# One-time network setup (routing + baseline QoS): run before any training
python setup_network.py --config prod.json
python setup_network.py --config prod.json --dry-run          # preview API calls only
python setup_network.py --config prod.json --continue-on-error

# Smoke test — hits Ryu telemetry, action, and env APIs; runs 3 env steps
python clint_test.py

# Train
python train.py --config prod.json --model-path models/dqn_model_live.pth
python train.py --config prod.json --skip-setup --seed 42    # skip startup setup
python train.py --config prod.json --resume models/train_state.pth   # continue after a controller outage

# Evaluate
python evaluate.py --config prod.json --model-path models/dqn_model_live.pth
python evaluate.py --config prod.json --skip-setup --metrics-path logs/eval.json

# Run reward-alignment batch experiment (multi-seed train+eval)
python run_reward_alignment_experiment.py   # writes configs/reward_alignment_exp.json

# Traffic generation (run inside Mininet)
python traffic_runner.py
```

Quick controller health checks:
```bash
curl http://<controller_ip>:8080/stats/switches
curl http://<controller_ip>:8080/links/utilization
curl http://<controller_ip>:8080/latency/<src>/<dst>
```

## Architecture

This project trains a DQN agent to optimize a live SDN network by issuing control commands to a Ryu controller over its REST API. There is **no simulation or mock mode** — every run requires a reachable Ryu controller.

### Two-phase execution contract

**Startup setup (one-time):** `NetworkInitializer` walks `prod.json:environment.startup_setup` and makes Ryu API calls to configure routing (addresses, static routes, default gateways on every switch) and baseline QoS queues/rules. Controlled by `environment.startup_setup.enabled` and the `--skip-setup` flag.

**Runtime (per RL step):** `SDNEnv.step()` executes one of 4 optimization-only actions, waits `stabilization_delay_seconds`, polls telemetry, builds a 6D state, and computes a reward.

### Data flow per step

```
ActionTranslator.execute(action_id)
    → RyuClient POST /qos/queue | /router | (no-op)
    → time.sleep(stabilization_delay)
    → RyuClient GET /links/utilization + GET /latency/{src}/{dst}
    → TelemetryParser.build_state(...)   # → np.float32[6]
    → compute_reward_details(state, config)
    → SDNEnv returns (state, reward, terminated, truncated, info)
```

### Key files

| File | Role |
|------|------|
| `prod.json` | Single source of truth: Ryu URL, DPIDs, state/action dims, hyperparameters, reward weights, startup setup payloads |
| `ai_layer/environments/sdn_env.py` | Gymnasium `Env` wrapping live telemetry and actions |
| `ai_layer/network_interface/ryu_client.py` | HTTP client with retry; normalizes decimal DPIDs to 16-hex for conf/router/qos endpoints |
| `ai_layer/network_interface/telemetry_parser.py` | Converts raw JSON into 6D normalized state |
| `ai_layer/network_interface/action_translator.py` | Maps action IDs to `RyuClient` calls; returns `ActionResult` |
| `ai_layer/network_setup/network_initializer.py` | One-time routing + QoS baseline setup |
| `ryu_apps/ovs_utilization.py` | Ryu app serving `GET /links/utilization`; owns the alias→DPID map (`name_map`) |
| `ryu_apps/network6G_monitor.py` | Ryu app serving `GET /latency/{src}/{dst}` |
| `ai_layer/agent/dqn_agent.py` | DQN with target network, epsilon-greedy, gradient clipping |
| `ai_layer/utils/reward.py` | Decomposed operational reward (latency/loss penalties, throughput bonus, congestion threshold) |

### Controller-side apps

`ryu_apps/` holds the two custom Ryu apps the telemetry pipeline depends on.
They are **not** importable Python for this project (they need `ryu` + `webob`,
neither in `requirements.txt`) — they run inside the Mininet/Ryu container and
must be copied to its `/root` before starting `ryu-manager`, which loads them by
bare relative path:

```bash
docker cp ryu_apps/network6G_monitor.py <container>:/root/
docker cp ryu_apps/ovs_utilization.py   <container>:/root/
```

Without them `/links/utilization` and `/latency/{src}/{dst}` do not exist, and
every telemetry read fails — `TelemetryParser` reads nothing else, and
`RyuClient.ping()` polls `/links/utilization` as its health check.
`ovs_utilization.py:36` is the single source of the alias→DPID map
(RAN=16, agg=32, core=48, sp1=64, sp2=65, lf1=80).

### State and action spaces

**State** (6D float32, all in [0, 1]):
`[latency_norm, packet_loss_norm, throughput_norm, main_link_util, backup_link_util, failover_active]`

**Actions** (Discrete 4):
- `0` do_nothing — no API call
- `1` update_queue — POST `/qos/queue/{dpid}`
- `2` failover — POST `/router/{dpid}` switching route to backup path, sets `failover_active=True`
- `3` reroute — POST `/router/{dpid}` restoring main path, sets `failover_active=False`

Episodes are **truncated** at `environment.episode.max_steps`; `terminated` is always `False`.

### DPID handling

`prod.json` stores DPIDs as decimal strings (e.g. `"48"`). `RyuClient._normalize_dpid()` converts to 16-character zero-padded hex before calling conf/router/qos endpoints. Telemetry endpoints (`/stats/switches`, `/stats/port/`) use raw decimal. Do not change DPIDs to hex in `prod.json`.

### Known failure modes

- **Controller unreachable**: update `environment.ryu_controller.base_url` in `prod.json` (default `http://172.17.0.2:8080`).
- **Latency returns null**: add default routes in Mininet host namespaces (`mnexec -a <pid> ip route add default via <gw>`).
- **qos_rest_router KeyError**: Mininet was started after Ryu — restart Ryu after Mininet is fully up.
- **`Address already in use` on ryu-manager start**: a stale controller still holds port 8080. Clear it before restarting: `fuser -k 8080/tcp` (or `kill $(lsof -t -i:8080)`), confirm with `ss -lntp | grep 8080`, then start `ryu-manager`. A Ryu restart also wipes the routing/QoS baseline — re-run `setup_network.py` (training does this automatically on reconnect).
- **Controller dies mid-training**: `train.py` saves `models/train_state.pth` every episode and waits `reconnect_max_wait_seconds` (default 180s) for the controller to return, re-applying startup setup before continuing. If it stays down, resume with `python train.py --config prod.json --resume models/train_state.pth`. Capture the cause by running the controller with its log teed to a file — see the full invocation in `docs/RUNNING_GUIDE.md` §3 (use `tee -a`; bare `tee` truncates the log on every restart).
- **Long training killed on disconnect**: run inside `tmux`/`screen` or use `nohup`.
- **Mininet `py` commands**: Mininet uses Python 2.7; use `execfile('/root/script.py', {'net': net})` instead of `exec(open(...))`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
