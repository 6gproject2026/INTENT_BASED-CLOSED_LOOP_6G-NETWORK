# AI Layer Specifications

**Project: Closed-Loop Intent-Based Network System**

This doc outlines the architecture and design logic for our Reinforcement Learning (RL) traffic optimization engine.

---

## 1. Overview

### 1.1 Purpose

The AI Layer implements a Deep Q-Network (DQN) agent that optimizes SDN traffic routing and rate limiting decisions. It learns to balance network load across links while minimizing packet loss.

### 1.2 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Systems                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Mininet   │    │ Ryu SDN     │    │    OpenFlow Switch   │ │
│  │ (Traffic)   │    │ Controller  │    │    (DPID: 00...01)  │ │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘ │
└─────────┼──────────────────┼─────────────────────┼─────────────┘
          │                  │                     │
          │                  │ REST API            │
          │                  │ (localhost:8080)    │
          │                  ▼                     │
┌─────────┼─────────────────────────────────────────────────────┐
│         │              AI Layer (This Project)                │
│         │                                                     │
│  ┌──────┴──────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │  Telemetry  │───▶│    State    │───▶│   DQN Agent     │    │
│  │   Parser    │    │   Builder   │    │  (5→64→64→5)    │    │
│  └─────────────┘    └─────────────┘    └────────┬────────┘    │
│                                                  │             │
│         ┌────────────────────────────────────────┘             │
│         ▼                                                      │
│  ┌─────────────────┐    ┌─────────────┐                        │
│  │    Action       │───▶│  REST Client│─────────────────────▶  │
│  │   Translator    │    │  (requests) │                        │
│  └─────────────────┘    └─────────────┘                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 AI Layer Responsibilities

The agent handles the following workflows:

| Duty | Description |
|----------------|-------------|
| Telemetry Parsing | Ingests and processes JSON stats from Ryu's REST API |
| State Construction | Converts raw data into our normalized 6-dimensional state vector |
| RL Decision Making | Picks actions using the DQN policy |
| Action Translation | Translates the numerical action into Ryu REST API calls |
| Training & Eval | Handles the episode loops, replay buffer writes, and weight updates |

### 1.4 Out of Scope

To keep separation of concerns, the AI Layer does **not** touch:
- Direct OpenFlow switch communication
- Ryu controller implementation or configuration
- Mininet topology management
- Traffic generation scripts

---

## 2. Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| RL Framework | PyTorch | ≥2.0 | Neural network, training loop |
| API Client | requests | ≥2.28 | HTTP communication with Ryu |
| Monitoring | TensorBoard | ≥2.12 | Training visualization |
| Configuration | JSON | — | Environment parameters |
| Environment Wrapper | gym | ≥0.26 | RL environment interface |

---

## 3. Telemetry Processing Pipeline

### 3.1 Data Flow

```
Ryu REST API → Telemetry Parser (pluggable) → State Vector Builder → Normalization → RL Agent
```

### 3.2 Expected Ryu Endpoints

| Endpoint | Response Format | Fields Used |
|----------|-----------------|--------------|
| `/stats/port/{dpid}` | Port statistics JSON | `rx_bytes`, `tx_bytes`, `rx_dropped`, `tx_dropped` |
| `/stats/flow/{dpid}` | Flow statistics JSON | Flow-specific metrics |
| `/qos/queue/{switch_id}` | Queue statistics | Queue lengths, drops |

### 3.3 Telemetry Parser Design (Pluggable)

The parser module should be designed for easy adjustment once controller responses are finalized:

```python
# Recommended interface
class TelemetryParser:
    """Pluggable parser for Ryu controller responses."""

    def parse_port_stats(self, response: dict) -> PortStatistics:
        """Extract port statistics from Ryu response."""
        # Implementation depends on actual Ryu response format
        pass

    def parse_flow_stats(self, response: dict) -> FlowStatistics:
        """Extract flow statistics from Ryu response."""
        pass

    def parse_queue_stats(self, response: dict) -> QueueStatistics:
        """Extract queue statistics from Ryu response."""
        pass
```

### 3.4 Expected Telemetry Fields

| Category | Fields | Notes |
|----------|--------|-------|
| Port Stats | `rx_packets`, `tx_packets`, `rx_bytes`, `tx_bytes` | Counters |
| Port Stats | `rx_dropped`, `tx_dropped` | Packet loss indicators |
| Flow Stats | Flow-specific metrics | TBD based on controller |
| Queue Stats | Queue depths, drops | TBD based on controller |

---

## 4. Action Execution Pipeline

### 4.1 Data Flow

```
RL Agent (action_id) → Action Translator → Ryu REST API Call → Network Change (handled by Ryu)
```

### 4.2 Action Definitions

| ID | Action Name | REST Call | Parameters |
|----|-------------|-----------|------------|
| 0 | `do_nothing` | None | — |
| 1 | `route_to_queue_0` | `POST /qos/rules/{switch_id}` | `queue_id=0`, protocol=UDP, port=5002, dst=10.0.0.1 |
| 2 | `route_to_queue_1` | `POST /qos/rules/{switch_id}` | `queue_id=1`, protocol=UDP, port=5002, dst=10.0.0.1 |
| 3 | `apply_rate_limit` | `POST /qos/queue/{switch_id}` | `port_name=s1-eth1`, `max_rate=500000` bps |
| 4 | `remove_rate_limit` | `POST /qos/queue/{switch_id}` | `port_name=s1-eth1`, `max_rate=1000000` bps |

### 4.3 Action Translator Interface

```python
class ActionTranslator:
    """Converts action IDs to Ryu REST API calls."""

    def __init__(self, ryu_client: RyuRestClient):
        self.client = ryu_client
        self.action_handlers = {
            0: self._do_nothing,
            1: self._route_to_queue_0,
            2: self._route_to_queue_1,
            3: self._apply_rate_limit,
            4: self._remove_rate_limit,
        }

    def execute(self, action_id: int) -> ActionResult:
        """Execute action and return result."""
        handler = self.action_handlers.get(action_id)
        if handler is None:
            raise ValueError(f"Invalid action_id: {action_id}")
        return handler()

    def _do_nothing(self) -> ActionResult:
        return ActionResult(success=True, message="No action taken")

    def _route_to_queue_0(self) -> ActionResult:
        return self.client.post_qos_rule(
            queue_id=0, protocol="UDP", port=5002, destination_ip="10.0.0.1"
        )

    # ... other action handlers
```

---

## 5. State Space Definition

### 5.1 Features (6-dimensional)

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | `latency` | Observed latency on primary test pair | min-max [10ms, 80ms] → [0,1] |
| 1 | `packet_loss` | Observed packet loss percentage | min-max [0%, 5%] → [0,1] |
| 2 | `throughput` | Total traffic across active links | min-max relative to total capacity |
| 3 | `main_link_util` | Utilization of the primary 20Mbps path | ratio [0,1] |
| 4 | `backup_link_util` | Utilization of the secondary 10Mbps path | ratio [0,1] |
| 5 | `failover_active` | Boolean flag indicating if failover is triggered | discrete {0.0, 1.0} |

### 5.2 Normalization Method

```python
def normalize_state(raw_state: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] range."""
    # State features are already normalized by Ryu (utilization as ratio)
    # Apply clipping for safety
    return np.clip(raw_state, 0.0, 1.0)
```

### 5.3 State Construction

```python
def build_state(port_stats: dict, flow_stats: dict) -> np.ndarray:
    """Construct 5-dim state vector from telemetry."""
    state = np.zeros(5, dtype=np.float32)

    # Link utilizations (from port stats)
    link_capacity = 1_000_000  # 1 Mbps from prod.json
    state[0] = port_stats['port_1']['tx_bytes'] / link_capacity
    state[1] = port_stats['port_2']['tx_bytes'] / link_capacity
    state[2] = port_stats['port_3']['tx_bytes'] / link_capacity

    # Packet loss rate
    total_packets = port_stats['port_1']['rx_packets'] + \
                    port_stats['port_2']['rx_packets'] + \
                    port_stats['port_3']['rx_packets']
    dropped = port_stats['port_1']['rx_dropped'] + \
              port_stats['port_2']['rx_dropped'] + \
              port_stats['port_3']['rx_dropped']
    state[3] = dropped / max(total_packets, 1)

    # Total traffic load
    state[4] = state[0] + state[1] + state[2]

    return normalize_state(state)
```

---

## 6. Action Space Definition

### 6.1 Discrete Actions

| Action | Network Effect | Use Case |
|--------|---------------|----------|
| `do_nothing` | No changes | Stable state, good performance |
| `route_to_queue_0` | UDP → high-priority queue | Latency-sensitive traffic |
| `route_to_queue_1` | UDP → low-priority queue | Bulk transfer, congestion relief |
| `apply_rate_limit` | Limit s1-eth1 to 500kbps | Congestion control |
| `remove_rate_limit` | Restore s1-eth1 to 1Mbps | Normal operation |

### 6.2 Action Constraints

- Actions are **mutually exclusive** per step
- `apply_rate_limit` and `remove_rate_limit` are **opposing** actions
- Rate limit actions affect only port `s1-eth1`

---

## 7. Reward Function

### 7.1 Multi-Objective Formula

```
R = R_util + R_loss + R_balance + R_congestion
```

### 7.2 Components

| Component | Formula | Weight | Description |
|-----------|---------|--------|-------------|
| Utilization Penalty | `R_util = -mean(u₁, u₂, u₃)²` | 1.0 | Penalize high utilization |
| Packet Loss Penalty | `R_loss = -packet_loss_rate × 100` | 100.0 | Heavy penalty for drops |
| Balance Bonus | `R_balance = 1 / (1 + std(u₁, u₂, u₃))` | 1.0 | Reward balanced load |
| Congestion Penalty | `R_cong = -5 if any(u) > 0.8 else 0` | 5.0 threshold | Extra penalty for congestion |

### 7.3 Implementation

```python
def compute_reward(state: np.ndarray) -> float:
    """Compute reward from normalized state vector."""
    u1, u2, u3, loss, total = state

    # Utilization penalty (quadratic)
    mean_util = (u1 + u2 + u3) / 3
    r_util = -(mean_util ** 2)

    # Packet loss penalty (heavy weight)
    r_loss = -loss * 100.0

    # Balance bonus (inverse std)
    util_std = np.std([u1, u2, u3])
    r_balance = 1.0 / (1.0 + util_std)

    # Congestion penalty
    r_congestion = -5.0 if max(u1, u2, u3) > 0.8 else 0.0

    # Total reward (clip to [-10, 10])
    reward = r_util + r_loss + r_balance + r_congestion
    return float(np.clip(reward, -10.0, 10.0))
```

---

## 8. DQN Agent Architecture

### 8.1 Neural Network

```
### 6.1 Discrete Actions

| Action ID | Network Effect | Description |
|-----------|----------------|-------------|
| `0` | `do_nothing` | Stable state, keep current routing and queue settings |
| `1` | `update_queue` | Adjust QoS queue bandwidth allocations on main core uplink to prioritize critical traffic |
| `2` | `failover` | Switch service route to backup path (spine2) when primary is congested |
| `3` | `reroute` | Restore service route back to main path (spine1) when conditions normalize | Layer:    5 neurons (state dimension)
Hidden Layer 1: 64 neurons, ReLU activation
Hidden Layer 2: 64 neurons, ReLU activation
Output Layer:   5 neurons (action Q-values)
```

### 8.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.001 | Adam optimizer |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Per-episode decay |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 100 | Steps between target network updates |

### 8.3 Replay Buffer

| Parameter | Value |
|-----------|-------|
| `capacity` | 10,000 transitions |
| `min_size_for_training` | 1,000 transitions |

### 8.4 Loss Function

- **MSE** (Mean Squared Error) for Q-value regression

---

## 9. Training vs Inference Modes

### 9.1 Training Mode

| Aspect | Behavior |
|--------|----------|
| Action Selection | ε-greedy (ε decays from 1.0 to 0.01) |
| Experience Replay | Enabled, samples from buffer |
| Gradient Updates | Every step after warmup |
| Model Updates | Target network synced every 100 steps |
| Checkpoints | Saved every 100 episodes |

### 9.2 Inference Mode

| Aspect | Behavior |
|--------|----------|
| Action Selection | Greedy (argmax Q(s, a)) |
| Experience Replay | Disabled |
| Gradient Updates | Disabled |
| Model Loading | Load best checkpoint |

### 9.3 Mode Switching

```python
class DQNAgent:
    def train_mode(self):
        self.training = True
        self.epsilon = self.epsilon_start

    def eval_mode(self):
        self.training = False
        self.epsilon = 0.0
```

---

## 10. Development Plan

### 10.1 Dependency Order

```
1. Ryu REST Client ──────────────────────────────────────────────────────────────┐
2. Telemetry Parser ──────────────────────────────────────────────────────────────┤
3. Action Translator ──────────────────────────────────────────────────────────────┤
4. Environment Wrapper ───────────────────────────────────────────────────────────┤
5. DQN Agent ──────────────────────────────────────────────────────────────────────┤
6. Training Loop ──────────────────────────────────────────────────────────────────┤
7. Evaluation Module ──────────────────────────────────────────────────────────────┘
```

### 10.2 Module Descriptions

#### Module 1: Ryu REST Client

**Purpose:** Base HTTP client for all Ryu API communication.

**Key Methods:**
```python
class RyuRestClient:
    def __init__(self, base_url: str, timeout: int, retry_attempts: int): ...
    def get_port_stats(self, dpid: str) -> dict: ...
    def get_flow_stats(self, dpid: str) -> dict: ...
    def post_qos_rule(self, switch_id: str, rule: dict) -> dict: ...
    def post_qos_queue(self, switch_id: str, queue: dict) -> dict: ...
    def reset_network(self) -> bool: ...
```

**Configuration:** `prod.json → environment.ryu_controller`

---

#### Module 2: Telemetry Parser

**Purpose:** Parse controller responses into structured data (pluggable design).

**Design Note:** JSON response structure is configurable. Parser should be easily adjustable once controller responses are finalized.

```python
@dataclass
class PortStatistics:
    rx_packets: int
    tx_packets: int
    rx_bytes: int
    tx_bytes: int
    rx_dropped: int
    tx_dropped: int

class TelemetryParser:
    def parse_port_stats(self, response: dict) -> Dict[int, PortStatistics]: ...
    def parse_flow_stats(self, response: dict) -> List[FlowEntry]: ...
```

---

#### Module 3: Action Translator

**Purpose:** Convert action IDs (0-4) to Ryu REST API calls.

```python
class ActionTranslator:
    def execute(self, action_id: int) -> ActionResult: ...
```

**Dependencies:** Module 1 (Ryu REST Client)

---

#### Module 4: Environment Wrapper

**Purpose:** Gym-style interface for RL training.

```python
class SDNEnvironment(gym.Env):
    def __init__(self, config: dict): ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]: ...
    def render(self, mode: str = "human"): ...
```

**Dependencies:** Modules 1, 2, 3

---

#### Module 5: DQN Agent

**Purpose:** Neural network and training logic.

```python
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict): ...
    def select_action(self, state: np.ndarray) -> int: ...
    def train_step(self, batch: List[Transition]) -> float: ...
    def update_target_network(self): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
```

**Dependencies:** Module 4 (for integration), PyTorch

---

#### Module 6: Training Loop

**Purpose:** Episode management, logging, checkpointing.

```python
class Trainer:
    def __init__(self, agent: DQNAgent, env: SDNEnvironment, config: dict): ...
    def train(self, num_episodes: int) -> List[float]: ...
    def evaluate(self, num_episodes: int) -> dict: ...
```

**Dependencies:** Modules 4, 5, TensorBoard

---

#### Module 7: Evaluation Module

**Purpose:** Metrics computation, baseline comparison.

```python
class Evaluator:
    def evaluate_policy(self, agent: DQNAgent, num_episodes: int) -> dict: ...
    def compare_baselines(self, agent: DQNAgent, baselines: List[Policy]) -> dict: ...
    def compute_metrics(self, trajectories: List[Trajectory]) -> dict: ...
```

**Dependencies:** Modules 4, 5

---

## 11. Traffic Generation Requirements

### 11.1 AI Layer Expectations

The AI Layer expects traffic to be generated by external Mininet environment. It does **NOT** create traffic.

### 11.2 Required Traffic Patterns

| Pattern | Type | Source | Bandwidth | Purpose |
|---------|------|--------|-----------|---------|
| `constant_load` | UDP | h2 | 800kbps | Baseline traffic |
| `bursty_load` | UDP | h3 | 1200kbps (burst) | Stress testing |
| `background` | ICMP | all | 1Hz ping | Connectivity |

### 11.3 Coordination with Network Engineer

- Traffic generation scripts run in Mininet environment
- AI Layer only observes and reacts
- Coordinate traffic timing with episode start/reset

---

## 12. Scenario Control

### 12.1 Training Scenarios

| Scenario | Description | Expected Learning |
|----------|-------------|-------------------|
| Normal Load | Traffic within capacity | Learn to balance |
| Congestion | Traffic exceeds capacity | Learn rate limiting |
| Burst | Sudden traffic spikes | Learn reactive routing |

### 12.2 Scenario Interface

```python
class ScenarioController:
    """Optional: Interface with external scenario trigger."""

    def load_scenario(self, name: str) -> bool: ...
    def get_current_scenario(self) -> str: ...
```

**Note:** Implementation depends on Network Engineer's scenario management system.

---

## 13. Evaluation Metrics

### 13.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Episode Reward | Cumulative reward per episode | Maximize |
| Packet Loss Rate | Average drop rate | < 5% |
| Utilization Balance | Std deviation of link usage | Minimize |
| Utilization | Mean link usage | < 80% |

### 13.2 Secondary Metrics

| Metric | Description |
|--------|-------------|
| Average Latency | End-to-end delay |
| Flow Completion Time | Time to complete flow |
| Action Distribution | Frequency of each action |
| Training Stability | Reward variance over time |

### 13.3 Baseline Comparison

| Baseline | Description |
|----------|-------------|
| Random | Random action selection |
| Do Nothing | Always select action 0 |
| Learned Policy | Trained DQN agent |

---

## 14. API Contracts

### 14.1 Ryu REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats/port/{dpid}` | Port statistics |
| GET | `/stats/flow/{dpid}` | Flow statistics |
| POST | `/qos/queue/{switch_id}` | Configure queue |
| POST | `/qos/rules/{switch_id}` | Add QoS rule |
| POST | `/network/reset` | Reset network state |

### 14.2 Request/Response Formats

**Note:** These formats are provisional. Coordinate with Network Engineer for exact specifications.

#### Port Statistics Response (Provisional)

```json
{
  "dpid": "0000000000000001",
  "ports": [
    {
      "port_no": 1,
      "rx_packets": 12345,
      "tx_packets": 67890,
      "rx_bytes": 1024000,
      "tx_bytes": 2048000,
      "rx_dropped": 10,
      "tx_dropped": 5
    }
  ]
}
```

#### QoS Rule Request

```json
{
  "queue_id": 0,
  "match": {
    "protocol": "UDP",
    "dst_port": 5002,
    "dst_ip": "10.0.0.1"
  }
}
```

### 14.3 Error Handling

```python
class RyuAPIError(Exception):
    """Base exception for Ryu API errors."""

class RyuConnectionError(RyuAPIError):
    """Failed to connect to Ryu controller."""

class RyuResponseError(RyuAPIError):
    """Invalid response from Ryu controller."""
```

---

## 15. File Structure (Recommended)

```
AI_Layer/
├── prod.json                 # Configuration file
├── PROJECT_SPECIFICATION.md  # This document
├── CLAUDE.md                 # Claude Code instructions
├── src/
│   ├── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   └── ryu_rest.py       # Module 1: REST client
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── telemetry.py      # Module 2: Telemetry parser
│   ├── actions/
│   │   ├── __init__.py
│   │   └── translator.py     # Module 3: Action translator
│   ├── env/
│   │   ├── __init__.py
│   │   └── sdn_env.py        # Module 4: Environment wrapper
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── dqn.py            # Module 5: DQN agent
│   │   └── replay_buffer.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py        # Module 6: Training loop
│   └── eval/
│       ├── __init__.py
│       └── evaluator.py      # Module 7: Evaluation
├── tests/
│   └── ...
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── runs/                     # TensorBoard logs
```

---

## 16. Configuration Reference

All values sourced from `prod.json`:

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| `environment.ryu_controller` | `base_url` | `http://localhost:8080` | Ryu API endpoint |
| `environment.ryu_controller` | `timeout_seconds` | `5` | Request timeout |
| `environment.network` | `switch_dpid` | `0000000000000001` | OpenFlow switch ID |
| `environment.network` | `link_capacity_bps` | `1000000` | 1 Mbps link capacity |
| `environment.network` | `stabilization_delay_seconds` | `2.0` | Delay after action |
| `environment.episode` | `max_steps` | `50` | Max steps per episode |
| `agent.hyperparameters` | `learning_rate` | `0.001` | Adam learning rate |
| `agent.hyperparameters` | `gamma` | `0.99` | Discount factor |
| `agent.hyperparameters` | `epsilon_decay` | `0.995` | Exploration decay |
| `training` | `num_episodes` | `1000` | Total training episodes |
| `training` | `warmup_steps` | `1000` | Steps before training |

---

## 17. Quick Start

### Development Workflow

1. **Start Ryu Controller** (Network Engineer responsibility)
2. **Start Mininet** with traffic generation
3. **Run AI Layer**:

```python
# Load configuration
config = load_config("prod.json")

# Initialize components
client = RyuRestClient(config["environment"]["ryu_controller"])
parser = TelemetryParser()
translator = ActionTranslator(client)
env = SDNEnvironment(client, parser, translator, config)
agent = DQNAgent(config)

# Training loop
trainer = Trainer(agent, env, config)
trainer.train(num_episodes=1000)
```

---

## Appendix A: Dependency Checklist

| Dependency | Required For | Install |
|------------|--------------|---------|
| PyTorch ≥2.0 | DQN Agent | `pip install torch` |
| gym ≥0.26 | Environment | `pip install gymnasium` |
| requests ≥2.28 | REST Client | `pip install requests` |
| TensorBoard ≥2.12 | Monitoring | `pip install tensorboard` |
| numpy | State vectors | `pip install numpy` |

---

## Appendix B: Troubleshooting

| Issue | Possible Cause | Solution |
|-------|-----------------|----------|
| Connection refused | Ryu not running | Start Ryu controller on localhost:8080 |
| Invalid state | Telemetry parse error | Check Ryu response format |
| Action failed | Ryu API error | Check QoS rule syntax |
| Training unstable | Learning rate too high | Reduce to 0.0001 |
| No improvement | Reward function issue | Verify reward computation |

---

*Document Version: 1.0.0*
*Last Updated: 2026-03-09*