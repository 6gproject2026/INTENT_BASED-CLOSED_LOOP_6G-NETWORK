# Closed-Loop Intent-Based 6G Network — Operator & Troubleshooting Guide

> **INSTRUCTIONS FOR THE ASSISTANT READING THIS FILE**
>
> You are supporting an operator running this project. This document is your complete
> ground truth — prefer it over general knowledge about Ryu, Mininet, or RL.
>
> Rules:
> 1. **There is no simulation mode.** Every run requires a live Ryu controller. "Just mock it"
>    is never a valid fix except for `test_intent_cli.py`, which is mock-only by design.
> 2. **Diagnose before prescribing.** Most reported failures are one of the 12 catalogued in
>    §8. Match the error signature first. If the report is ambiguous, ask for the specific
>    diagnostic output listed in §9 — don't guess.
> 3. **Order of operations is the #1 cause of failures here.** Before anything else, confirm the
>    operator followed §3 exactly. Mininet must be fully up *before* Ryu starts.
> 4. **Do not suggest changing DPIDs to hex in `prod.json`.** See §6.4 — the conversion is
>    automatic and doing it manually breaks telemetry.
> 5. **Do not suggest lowering `num_episodes` without also retuning `epsilon_decay`.** They are
>    mathematically coupled. See §5.2 — this silently ruins training runs.
> 6. Give the exact command or exact file+line edit. Avoid "you might try" lists.

---

## 1. What this project is

A DQN reinforcement-learning agent that optimizes a live software-defined network by issuing
control commands to a Ryu SDN controller over its REST API. A natural-language **intent layer**
sits alongside it, compiling operator sentences ("Cap mMTC traffic to 5 Mbps on switch 0x30")
into the same controller calls.

Two independent control paths write to the same controller:

```
RL path:      SDNEnv.step(action_id) -> ActionTranslator -> RyuClient -> Ryu REST
Intent path:  "cap mMTC to 5 Mbps"  -> IntentParser -> PolicyGuard -> IntentTranslator -> RyuClient -> Ryu REST
```

Both funnel through `RyuClient`, which is the single chokepoint for HTTP, retries, and DPID
normalization.

---

## 2. Machine layout and versions

This runs on **two different Python environments**. Confusing them causes a large share of errors.

| Component | Where | Python | Notes |
|---|---|---|---|
| Ryu controller | Docker container (e.g. `root@debc86f1904f`) | **2.7** | `ryu==4.9`, eventlet WSGI on port 8080 |
| Mininet | Same host/container as Ryu | **2.7** | OVS switches |
| AI layer (this repo) | Host, e.g. `netadmin@mininet-ryu` | **3.10** | in `venv/` |

Controller reached at **`http://172.17.0.2:8080`** (the Docker bridge IP), configured in
`prod.json` → `environment.ryu_controller.base_url`.

Key AI-layer pins (`requirements.txt`): `torch==2.11.0`, `gymnasium==1.2.3`, `numpy==2.2.6`,
`requests==2.33.1`.

> Mininet's CLI is Python 2.7. Inside `mininet>` use
> `py execfile('/root/script.py', {'net': net})` — **not** `exec(open(...))`, which is Py3 syntax.

---

## 3. Startup order (follow exactly)

Order matters. Starting Ryu before Mininet causes `qos_rest_router` `KeyError` crashes.

```bash
# 1. Start Mininet FIRST and let the topology fully come up.
#    Wait until all switches/hosts are listed and pingable.

# 2. THEN start Ryu, capturing its log (needed for post-mortem diagnosis):
ryu-manager <your apps> > /tmp/ryu.log 2>&1 &

# 3. Verify the controller answers before doing anything else:
curl http://172.17.0.2:8080/stats/switches
curl http://172.17.0.2:8080/links/utilization
curl http://172.17.0.2:8080/latency/G6_D1/URLLC

# 4. One-time network setup (routing + baseline QoS):
python setup_network.py --config prod.json

# 5. Smoke test the AI layer end-to-end:
python clint_test.py

# 6. Train:
python train.py --config prod.json --model-path models/dqn_model_live.pth
```

**Always run long training inside `tmux` or `screen`** — an SSH disconnect kills it otherwise.

### Restarting Ryu

A Ryu restart **wipes the routing and QoS baseline**. After any restart you must re-run
`setup_network.py` (training does this automatically on auto-reconnect — see §7).

```bash
fuser -k 8080/tcp          # clear the stale listener FIRST
ss -lntp | grep 8080       # must print nothing
ryu-manager <apps> > /tmp/ryu.log 2>&1 &
python setup_network.py --config prod.json
```

---

## 4. Command reference

```bash
# Setup
python setup_network.py --config prod.json
python setup_network.py --config prod.json --dry-run           # preview API calls only
python setup_network.py --config prod.json --continue-on-error

# Smoke tests
python clint_test.py                    # telemetry + action + env, 3 env steps
python smoke_test_live.py               # intent layer against LIVE Ryu

# Train
python train.py --config prod.json --model-path models/dqn_model_live.pth
python train.py --config prod.json --skip-setup --seed 42
python train.py --config prod.json --resume models/train_state.pth   # after an outage

# Evaluate
python evaluate.py --config prod.json --model-path models/dqn_model_live.pth
python evaluate.py --config prod.json --skip-setup --metrics-path logs/eval.json

# Intent layer
python run_intent.py "Cap mMTC traffic to 5 Mbps on switch 0x30"      # LIVE
python run_intent.py "Show utilization" --dry-run                      # prints calls, sends nothing
python test_intent_cli.py "Cap mMTC traffic to 5 Mbps on switch 0x30"  # MOCK ONLY, never touches network

# Batch experiment
python run_reward_alignment_experiment.py

# Traffic generation (run INSIDE Mininet)
python traffic_runner.py
```

---

## 5. Configuration (`prod.json` is the single source of truth)

### 5.1 Controller / resilience block

```json
"environment": { "ryu_controller": {
  "base_url": "http://172.17.0.2:8080",
  "timeout_seconds": 15,
  "retry_attempts": 6,
  "retry_delay_seconds": 1,
  "retry_backoff_max_seconds": 8,
  "reconnect_max_wait_seconds": 180,
  "reconnect_poll_seconds": 5
}}
```

Per-request retry uses capped exponential backoff: delays `1,2,4,8,8` = **23 s** of tolerance.
The longer 180 s wait is the training-loop reconnect (§7).

### 5.2 ⚠ Episode budget and epsilon are coupled — read before changing either

`agent.decay_epsilon()` is called **once per environment step**, not per episode. So:

```
steps_to_reach_epsilon_min = log(epsilon_min) / log(epsilon_decay)
total_steps                = num_episodes * max_steps
```

Current settings: `num_episodes: 100`, `max_steps: 50` → 5,000 steps;
`epsilon_decay: 0.9991`, `epsilon_start: 1.0`, `epsilon_end: 0.01` → final epsilon **0.011** ✓

| num_episodes | total steps | required epsilon_decay |
|---|---|---|
| 100 | 5,000 | **0.9991** |
| 200 | 10,000 | 0.99954 |
| 300 | 15,000 | 0.99969 |

**Failure mode:** setting `num_episodes: 100` while leaving `epsilon_decay: 0.9996` leaves epsilon
stuck at **0.135** — the agent stays 13.5 % random forever and never converges. The run *looks*
fine; rewards just never improve. If an operator reports "training finished but the agent is bad"
or "reward is flat", **check this ratio first**.

Verify with:
```bash
python -c "print(0.9991**5000)"   # expect ~0.011
```

### 5.3 Timing / expected wall-clock

Per step: `stabilization_delay_seconds` (1.0 s) + 2 telemetry GETs + up to 1 action POST.

```
wall_clock ≈ num_episodes * (max_steps + 1) * stabilization_delay
100 eps × 51 × 1.0 s ≈ 1.4 h   (sleep only; add HTTP overhead)
```

If an operator says "it's taking hours", check `num_episodes` and `max_steps` before anything else.
Lowering `stabilization_delay_seconds` speeds things up but gives the network less time to settle
after a QoS change, making telemetry noisier.

### 5.4 Other key blocks

- `environment.network` — `main_link_capacity_mbps: 20.0`, `backup_link_capacity_mbps: 10.0`,
  `switch_dpid: "48"` (core).
- `environment.monitoring.main_pair` — `{src: "G6_D1", dst: "URLLC"}`, used for `/latency/{src}/{dst}`.
  `latency_min_ms: 10.0`, `latency_max_ms: 80.0`, `packet_loss_max_percent: 5.0`.
- `environment.episode` — `max_steps: 50`, `call_network_reset_on_reset: false`.
- `agent.hyperparameters` — `lr 0.001`, `gamma 0.99`, `batch_size 64`, `target_update_frequency 100`.
- `agent.replay_buffer` — `capacity 10000`, `min_size_for_training 200`.
- `training` — `num_episodes 100`, `warmup_steps 200`, `save_frequency 25`, `run_startup_setup true`.
- `system` — `random_seed 42`, `device "cpu"`.

---

## 6. Data contracts

### 6.1 Topology

```
main path:    core -> sp1 -> lf1        (20 Mbps)
backup path:  core -> sp2 -> lf1        (10 Mbps)
```

DPIDs (decimal, as stored in `prod.json`): `RAN 16, agg 32, core 48, sp1 64, sp2 65, lf1 80`.

### 6.2 State (6D float32, every element clipped to [0,1])

| idx | feature | derivation |
|---|---|---|
| 0 | `latency_norm` | `(latency_ms - 10) / (80 - 10)` |
| 1 | `packet_loss_norm` | `loss_pct / 5.0` |
| 2 | `throughput_norm` | `(core->sp1 + core->sp2) / 30 Mbps` |
| 3 | `main_link_util` | mean of `core -> sp1`, `sp1 -> lf1` ÷ 20 |
| 4 | `backup_link_util` | mean of `core -> sp2`, `sp2 -> lf1` ÷ 10 |
| 5 | `failover_active` | 1.0 / 0.0 |

**Important:** if `/latency` returns `null` for latency or loss, the parser substitutes
**worst-case** values (80 ms, 5 %), not zero. Symptom: state[0] and state[1] pinned at 1.0 and
reward stuck deeply negative. That means telemetry is broken, not that the network is bad. See §8.

Link names in `/links/utilization` must match **exactly**: `"core -> sp1"`, `"sp1 -> lf1"`,
`"core -> sp2"`, `"sp2 -> lf1"` (spaces around `->`). A naming mismatch silently yields 0.0
utilization for every link.

### 6.3 Actions (Discrete 4)

| id | name | call |
|---|---|---|
| 0 | `do_nothing` | none |
| 1 | `update_queue` | `POST /qos/queue/{dpid}` on core-eth2, linux-htb, max 20 Mbps |
| 2 | `failover` | `POST /router/{dpid}` route `20.0.0.0/24` via `14.0.0.2`; sets `failover_active=True` |
| 3 | `reroute` | `POST /router/{dpid}` route `20.0.0.0/24` via `13.0.0.2`; sets `failover_active=False` |

Episodes are **truncated** at `max_steps`; `terminated` is always `False`.

### 6.4 DPID normalization — do not "fix" this

`prod.json` stores DPIDs as **decimal strings** (`"48"`). `RyuClient._normalize_dpid()` converts to
16-char zero-padded hex (`0000000000000030`) automatically for `conf` / `router` / `qos` endpoints.
Telemetry endpoints (`/stats/switches`, `/stats/port/`) use **raw decimal**.

**Never convert DPIDs to hex in `prod.json`.** Doing so breaks the telemetry endpoints.

### 6.5 Endpoints used

```
GET  /stats/switches                GET  /links/utilization
GET  /stats/port/{dpid}             GET  /latency/{src}/{dst}
GET  /stats/flow/{dpid}             GET  /qos/queue/{dpid}
POST /qos/queue/{dpid}              POST /qos/rules/{dpid}
POST /router/{dpid}                 POST /stats/flowentry/add
POST /v1.0/conf/switches/{dpid}/ovsdb_addr
POST /network/reset
```

### 6.6 Reward (`ai_layer/utils/reward.py`)

```
latency_penalty      = -latency_norm * 2.5
packet_loss_penalty  = -loss_norm * 3.0
utilization_penalty  = -(mean_util^2) * 1.2
throughput_bonus     = +throughput_norm * 1.5
congestion_penalty   = -2.0  if max(main_util, backup_util) > 0.9
failover_penalty     = -failover_active * 0.2
action_repeat_penalty= -0.1  if action == previous action
outcome_bonus        = +min(0.2, congestion_improvement * 1.0)  for actions 1/2/3
total clipped to [-10, 10]
```

Rewards are **normally negative** — that is expected, not a bug. Judge progress by the trend of
`AvgReward(20)`, not the sign.

---

## 7. Fault tolerance (what training does on its own)

`train.py` handles controller outages without operator action:

1. **Preflight** — before training starts, waits up to 10 s for the controller. Fails fast with a
   clear message instead of dying mid-run.
2. **Per-request backoff** — `RyuClient` retries 6× with delays `1,2,4,8,8` (23 s).
3. **Ride-out** — if `RyuConnectionError` still escapes, training saves state, waits up to
   **180 s** for the controller to return, **re-runs `NetworkInitializer`** (the restart wiped
   routing/QoS), discards the interrupted episode, and continues.
4. **Resume** — `models/train_state.pth` is written **every episode**. If the controller stays down
   past 180 s, training exits cleanly (code 1) and prints the exact resume command.

```bash
python train.py --config prod.json --resume models/train_state.pth
```

Resume restores q-network, target network, optimizer, epsilon, episode index, global step,
reward history, and seed.

Two behaviours to expect on resume:
- **`--seed` is ignored** — the seed is restored from the file to keep per-episode seeding consistent.
- **The replay buffer is NOT saved** — it restarts empty and must refill to 200 transitions
  (~4 episodes) before gradient updates resume. Weights and epsilon carry over intact.

A recovered run finishes with 99 completed episodes rather than 100, because the interrupted
episode is deliberately discarded (its trajectory spans a network-state discontinuity).

---

## 8. Error catalogue — signature → cause → fix

### 8.1 `socket.error: [Errno 98] Address already in use` (from `ryu-manager`)

```
File ".../ryu/lib/hub.py", line 117, in __init__
    self.server = eventlet.listen(listen_info)
socket.error: [Errno 98] Address already in use
```

**Cause:** a previous Ryu process still holds port 8080. This is a *startup* failure — it is the
**consequence** of an earlier crash, not the original problem. Always ask what happened *before* this.

**Fix:**
```bash
fuser -k 8080/tcp
ss -lntp | grep 8080        # must be empty
ryu-manager <apps> > /tmp/ryu.log 2>&1 &
python setup_network.py --config prod.json   # restart wiped the baseline
```

### 8.2 `ConnectionRefusedError: [Errno 111]` / `RyuConnectionError: Cannot reach Ryu at ...`

**Cause:** nothing listening on 8080 — Ryu is down, or `base_url` is wrong.

**Fix:** verify `curl http://172.17.0.2:8080/stats/switches`. If refused, Ryu is dead → §8.1.
If the container IP changed, update `environment.ryu_controller.base_url`. Get the real IP with
`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container>`.

### 8.3 `ConnectionResetError(104, 'Connection reset by peer')` then repeated `Connection refused`

**Cause:** Ryu died *mid-request*. The reset is the socket breaking; the refusals follow because the
process is gone. This is the signature of a controller crash during a long run.

**Fix:** training now rides this out automatically (§7). If it exceeded 180 s, restart Ryu (§8.1)
and `--resume`. **To find the root cause, you need `/tmp/ryu.log`** — ask for it. Without Ryu's own
stderr the cause is not determinable from the Python side.

Contributing load: ~50 steps/episode × ~75 % write actions means thousands of `POST /qos/queue`
and `/router` calls per run. Repeated QoS posts create OVS QoS/Queue rows and are a plausible
pressure source on Ryu 4.9 / Python 2.7 / eventlet.

### 8.4 `/latency` returns `null`

**Cause:** hosts lack default routes in their Mininet namespaces.

**Fix:** inside Mininet, for each host: `mnexec -a <pid> ip route add default via <gw>`.
Confirm with `curl http://172.17.0.2:8080/latency/G6_D1/URLLC`.

**Why it matters:** the parser substitutes worst-case (80 ms, 5 %) for nulls, so training will run
but the agent learns from garbage. Symptom: state[0]/state[1] pinned at 1.0, reward deeply negative
and flat.

### 8.5 `qos_rest_router` `KeyError`

**Cause:** Mininet was started *after* Ryu, so the controller never registered the datapaths.

**Fix:** restart Ryu after Mininet is fully up (§3). Order is not optional.

### 8.6 Training runs for hours

**Cause:** `num_episodes` too high. 300 episodes ≈ 4.25 h; 100 ≈ 1.4 h.

**Fix:** set `num_episodes: 100` **and** `epsilon_decay: 0.9991` together (§5.2).

### 8.7 Training finishes but the agent never improves / reward is flat

**Cause (most common):** epsilon never annealed — `num_episodes` was reduced without retuning
`epsilon_decay`, leaving epsilon ~0.135 (§5.2). Check the `Epsilon:` value in the last episode line;
it should be ~0.01.

**Other causes:** all-null telemetry (§8.4); link-name mismatch making utilization always 0.0 (§6.2).

### 8.8 `State dimension mismatch` / `Action dimension mismatch`

**Cause:** `agent.neural_network.input_dim`/`output_dim` disagree with the env (6 and 4).

**Fix:** set `input_dim: 6`, `output_dim: 4` in `prod.json`.

### 8.9 Intent returns `"status": "success"` but nothing changed on the network

**Cause:** you ran `test_intent_cli.py`, which uses `MockRyuClient` and **never makes HTTP calls**
by design. It validates parsing/translation only.

**Fix:** use the live runner:
```bash
python run_intent.py "Cap mMTC traffic to 5 Mbps on switch 0x30"
```
Verify in three layers — controller 200 alone is not proof:
1. Ryu log shows `POST /qos/queue/0000000000000030 ... 200`
2. In Mininet/OVS: `ovs-vsctl list qos` and `ovs-vsctl list queue` show the new rate
3. Behaviour: `iperf` to the slice host caps at the requested rate

### 8.10 Intent targeting a non-existent switch reports success

**Known gap.** `IntentTranslator._handle_cap_traffic` passes the DPID straight through with no
allowlist check, and `resolve()` reports `"success"` whenever no exception was raised. Under the
mock client, a bogus DPID like `0x999` therefore "succeeds". Against live Ryu it will surface as
`"status": "error"`, but only after the call is attempted.

**Fix (not yet implemented):** add a DPID allowlist guard in `intent_layer/policy_guard.py` that
rejects DPIDs outside `prod.json` → `environment.network.switch_dpids` before any Ryu call.

### 8.11 Intent rejected: `unsupported` / `blocked`

- `unsupported` — the parser found no known action verb, or validation failed. `cap_traffic`
  requires **both** a rate and a slice; `reroute` requires a DPID. See §10 for accepted phrasings.
- `blocked` — `PolicyGuard` conflict: a higher-priority slice holds a conflicting write intent
  (`URLLC > eMBB > mMTC`), TTL 300 s. Read intents never block.

### 8.12 `ModuleNotFoundError: No module named 'torch'`

**Cause:** the venv isn't activated, or you're on the wrong machine/interpreter. Note the repo may
contain a **Windows** venv (`venv/Scripts/`) that is unusable from Linux/WSL.

**Fix:** `source venv/bin/activate && pip install -r requirements.txt`. Confirm with
`python -c "import torch; print(torch.__version__)"` → `2.11.0`.

---

## 9. Diagnostics to request when the report is ambiguous

Ask for whichever apply — do not guess without them.

```bash
# A. Is the controller alive and what does it see?
curl -s http://172.17.0.2:8080/stats/switches
curl -s http://172.17.0.2:8080/links/utilization
curl -s http://172.17.0.2:8080/latency/G6_D1/URLLC

# B. Ryu's own log — REQUIRED for any controller-crash question
tail -100 /tmp/ryu.log

# C. Port ownership (for 'Address already in use')
ss -lntp | grep 8080

# D. Effective config
python -c "import json;c=json.load(open('prod.json'));print(json.dumps({
 'base_url':c['environment']['ryu_controller']['base_url'],
 'num_episodes':c['training']['num_episodes'],
 'max_steps':c['environment']['episode']['max_steps'],
 'epsilon_decay':c['agent']['hyperparameters']['epsilon_decay'],
 'stabilization':c['environment']['network']['stabilization_delay_seconds']},indent=1))"

# E. Did QoS actually apply?
ovs-vsctl list qos
ovs-vsctl list queue

# F. Last training output (epsilon + reward trend)
tail -20 <training log>
```

**Triage order:** controller reachable? → telemetry non-null? → config sane? → then application logic.

---

## 10. Intent layer reference

**Actions:** `cap_traffic`, `prioritize_slice`, `reroute`, `failover`, `get_latency`, `get_utilization`
**Slices:** `URLLC`, `eMBB`, `mMTC`
**Queue map:** URLLC→1, eMBB→2, mMTC→3   **DSCP:** URLLC 46, eMBB 34, mMTC 10

Trigger words (regex, case-insensitive):

| Action | Keywords |
|---|---|
| `cap_traffic` | cap, limit, restrict |
| `prioritize_slice` | prioritize, prefer, boost |
| `reroute` | reroute, redirect, route |
| `failover` | failover, fail over |
| `get_latency` | latency, ping, delay |
| `get_utilization` | utilization, utilisation, throughput, bandwidth |

Rates: `5 Mbps`, `10mbps`, `100 kbps`, `1 Gbps`. DPIDs: `0x30`, `switch 48`, `dpid 0x1a`.
Defaults: DPID `0x30`, latency dst `0x40`.

Working examples:
```
"Cap mMTC traffic to 5 Mbps on switch 0x30"
"Prioritize URLLC on switch 0x30"
"Show latency between 0x30 and 0x40"
"Get utilization for switch 0x30"
```

Result statuses: `success` | `unsupported` (parse failed) | `blocked` (policy conflict) |
`error` (handler raised — includes unreachable Ryu).

---

## 11. File map

| File | Role |
|---|---|
| `prod.json` | Single source of truth: URL, DPIDs, dims, hyperparameters, reward weights, startup payloads |
| `train.py` | Training loop, preflight, ride-out reconnect, resume |
| `evaluate.py` | DQN vs random/do-nothing baselines |
| `setup_network.py` | One-time routing + QoS baseline |
| `clint_test.py` | Smoke test: telemetry, actions, 3 env steps |
| `smoke_test_live.py` | Intent layer against live Ryu |
| `run_intent.py` | Resolve one intent against **live** Ryu (`--dry-run` available) |
| `test_intent_cli.py` | Intent layer against a **mock** — never touches the network |
| `traffic_runner.py` | iperf traffic generation (run inside Mininet) |
| `ai_layer/environments/sdn_env.py` | Gymnasium env over live telemetry |
| `ai_layer/network_interface/ryu_client.py` | HTTP client: retry/backoff, DPID normalization, `ping`, `wait_for_controller` |
| `ai_layer/network_interface/telemetry_parser.py` | Raw JSON → 6D state |
| `ai_layer/network_interface/action_translator.py` | Action id → RyuClient calls |
| `ai_layer/network_setup/network_initializer.py` | Routing + QoS baseline setup |
| `ai_layer/agent/dqn_agent.py` | DQN: target network, epsilon-greedy, grad clipping |
| `ai_layer/utils/reward.py` | Decomposed operational reward |
| `intent_layer/intent_parser.py` | NL → `ParsedIntent` (regex) |
| `intent_layer/policy_guard.py` | Slice priority, TTL, conflict blocking |
| `intent_layer/intent_translator.py` | `ParsedIntent` → RyuClient calls |

---

## 12. Quick sanity checklist

- [ ] Mininet started **before** Ryu
- [ ] `curl .../stats/switches` returns switches
- [ ] `curl .../latency/G6_D1/URLLC` returns non-null latency **and** loss
- [ ] `setup_network.py` ran after the most recent Ryu start
- [ ] `base_url` matches the container IP
- [ ] DPIDs in `prod.json` are decimal
- [ ] `num_episodes` × `max_steps` matches `epsilon_decay` (§5.2)
- [ ] `input_dim: 6`, `output_dim: 4`
- [ ] Training running under `tmux`/`screen`
- [ ] Ryu started with `> /tmp/ryu.log 2>&1`
