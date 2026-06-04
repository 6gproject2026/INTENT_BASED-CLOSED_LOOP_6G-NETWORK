# OpenCode Agent Notes

## High-signal commands
- Install deps: `pip install -r requirements.txt`
- One-time network setup (routing + baseline QoS): `python setup_network.py --config prod.json`
  - Add `--dry-run` to print planned API calls, `--continue-on-error` to keep going on failures.
- Live smoke run (hits Ryu APIs, runs 3 env steps): `python clint_test.py`
- Train: `python train.py --config prod.json --model-path models/dqn_model_live.pth` (add `--skip-setup` to avoid startup setup)
- Evaluate: `python evaluate.py --config prod.json --model-path models/dqn_model_live.pth` (add `--skip-setup` to avoid startup setup)
- Reward-alignment batch: `python run_reward_alignment_experiment.py` (writes `configs/reward_alignment_exp.json`, runs train+eval for multiple seeds)

## Runtime expectations
- Most scripts call the live Ryu controller defined in `prod.json` (`environment.ryu_controller.base_url`); runs will fail without a reachable controller.
- Startup setup is controlled by config (`environment.startup_setup.enabled`) and can be skipped per-run with `--skip-setup`.

## Structure to know
- Main config file: `prod.json` (state/action sizes, Ryu endpoints, startup setup, reward config, episode limits).
- Live Gym environment: `ai_layer/environments/sdn_env.py` (telemetry + action loop, stabilization delay, reward computation).
- One-time setup logic: `ai_layer/network_setup/network_initializer.py` (routing/QoS baseline calls).
