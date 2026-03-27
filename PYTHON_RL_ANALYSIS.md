# Python RL Repo — Architecture Analysis
_Generated 2026-03-24. Mental model: external runtime is authoritative in live mode; Python mirrors live state into a shadow env for RL logic._

---

## 1. Repo Purpose

Trains a SAC agent to perform autonomous highway lane changes. Wraps HighwayEnv into a custom `LaneChangingEnv` that supports two backends:

- **Offline** (`backend="csv"`): HighwayEnv's built-in physics simulate everything. CSV trajectory data seeds initial ego state.
- **Live** (`backend="aimsun_live"`): External runtime (Aimsun Next / CDA) is the authoritative simulator. Python mirrors live state into HighwayEnv objects solely to run reward and observation logic.

---

## 2. Key Files

| File | Role |
|---|---|
| `src/sac/lanechange_env.py` | Core Gymnasium env — step, reset, reward, action routing |
| `src/sac/live_bridge.py` | `HighwayShadowBridge` — snapshot normalization, shadow state sync |
| `src/sac/wrappers.py` | `ObsWrapper` — flattens obs + appends lane-goal scalars |
| `src/sac/datalogger_reader.py` | Parses Aimsun 29-col HIL trajectory `.txt` files for offline seeding |
| `src/sac/thirdparty_cleanrl/sac_core.py` | CleanRL SAC training loop (largely unmodified) |

---

## 3. Key Classes / Functions

### `LaneChangingEnv` (`lanechange_env.py`)
- `step(action)` — branches on `backend` flag; two entirely separate code paths
- `reset()` — same branch; in live mode calls connector then syncs shadow state
- `_send_action(action)` — in live mode, stores decoded command only; does NOT apply physics
- `_apply_action(action)` — in offline mode, sets `vehicle.target_lane_index` and `vehicle.action["acceleration"]` directly
- `_reward(action)` — 7-term weighted reward built from `RewardTerm` descriptors; reads mirrored `env.vehicle.*` and `road.vehicles`
- `_get_surrounding_vehicles()` — queries `road.vehicles` for front/rear/llead/llag/rlead/rlag relative to ego
- `_check_lane_change_termination()` — terminates when `steps_in_target_lane >= duration_after_lane_change`

### `HighwayShadowBridge` (`live_bridge.py`)
- `require_connector()` — validates a connector is attached; raises `RuntimeError` otherwise
- `decode_action_command(action)` — maps `[accel_norm, intent]` → structured command dict (accel_mps2, target_lane_index, dt, desired_lane_id)
- `sync_shadow_state(snapshot)` — normalizes snapshot → writes into `env.vehicle.*`, rebuilds `env.road.vehicles`, sets `env.time`
- `_normalize_snapshot(snapshot)` — accepts both structured (`"ego"` key) and flat Aimsun format; outputs canonical normalized dict
- `_synthesize_surrounding_neighbors()` — builds shadow Vehicles from gap/speed data when no explicit `neighbors` list is present
- `_make_shadow_vehicle()` — creates a stateless `highway_env.Vehicle` object at the specified lane/position/speed

### `LiveConnector` (`live_bridge.py`) — canonical interface
- `reset_episode(seed) -> dict`
- `step(command: dict) -> dict`
- `close()`

> **Note**: A duplicate `LiveConnector` that existed in `lanechange_env.py` has been removed. The import from `live_bridge` is now the single definition.

### `ObsWrapper` (`wrappers.py`)
- Flattens HighwayEnv's `(vehicles_count, features)` matrix to 1D
- Appends 2 goal scalars: `lane_delta_to_target`, `target_lane_norm` (both clipped to `[-1, 1]`)

---

## 4. Data Flow

### Offline mode
```
SAC actor → action [accel_norm, intent]
  → LaneChangingEnv.step()
  → super().step()  [HighwayEnv runs physics]
  → _reward() reads env.vehicle.* + road.vehicles
  → observation_type.observe()
  → ObsWrapper appends goal scalars
  → SAC replay buffer
```

### Live mode
```
SAC actor → action [accel_norm, intent]
  → LaneChangingEnv.step()
  → decode_action_command()           → command dict
  → LiveConnector.step(command)       → raw snapshot  [EXTERNAL RUNTIME]
  → HighwayShadowBridge.sync_shadow_state(snapshot)
      _normalize_snapshot()           → canonical dict
      writes env.vehicle.{position, speed, lane_index, crashed, action["acceleration"]}
      writes env.road.vehicles        (ego + shadow neighbors)
      writes env.time
  → _reward()                         reads mirrored state
  → _observe_env() / observe()        reads mirrored state
  → ObsWrapper appends goal scalars
  → SAC replay buffer
```

### Reset (live)
```
LaneChangingEnv.reset()
  → LiveConnector.reset_episode(seed) → snapshot
  → sync_shadow_state(snapshot)
  → _reset_episode_state()            clears jerk/lane-change counters, sets target lane
  → _observe_env()
```

---

## 5. Assumptions This Repo Makes

1. Road is a single straight segment with `lanes_count` parallel lanes, uniform width — no intersections, curves, or merges.
2. Lane IDs are 0-based internally; Aimsun sends 1-based (bridge corrects with `lane - 1`).
3. Ego vehicle is always `env.vehicle` (index 0 in `road.vehicles`). Bridge replaces `road.vehicles` wholesale each step.
4. External runtime controls vehicle physics in live mode; HighwayEnv physics only matter in offline mode.
5. Policy frequency is 15 Hz (config default); `dt = 1/15` baked into `decode_action_command`.
6. Target lane is always an adjacent lane (`lane_id ± 1`). Multi-lane planning deferred ("A separate network will handle this later").
7. Speed limit is optional; absent → `_r_fn_matching_speed_limit` returns 0.
8. `env.time` is float seconds; jerk = `(curr_acc - prev_acc) / dt`.
9. CDA data is an opaque passthrough dict — bridge forwards `snapshot["cda"]` to `info["cda"]` without interpreting it.
10. Neighbor shadow vehicles are stateless for exactly one step (no IDM, no lane-change controllers on them).

---

## 6. What This Repo Expects From Other Repos

### From Aimsun/HIL plugin (offline CSV path)
A 29-column space-delimited `.txt` file written by `writeTestData2File()` in `AAPI.cxx`. Column order per `datalogger_reader._COL`:

| Col | Field | Notes |
|---|---|---|
| 0 | vehID | int |
| 1–4 | hour, min, sec, ms | int |
| 5 | simStep | int |
| 6–8 | testVehID, tpStatus, vehType | int |
| 9–10 | linkID, nodeID | int |
| 11 | lane | int, **1-based** |
| 12 | speed_mps | float, m/s |
| 13 | pos_m | float, m along link |
| 14 | totalDist | float |
| 15–17 | sysEntryStep, linkEntryStep, driveMode | int |
| 18 | lCDir | int (1=left, 2=none, 3=right) |
| 19–26 | lleadV/Gap, llagV/Gap, rleadV/Gap, rlagV/Gap | float |
| 27–28 | frontV, frontGap | float |

**Gap in CSV**: No rear same-lane vehicle columns (`rearV`, `rearGap`). In offline mode this is irrelevant (rear is computed from `road.vehicles` dynamically). In live mode with the flat Aimsun format, rear gap would be blind. Since the live connector uses the structured `"ego"` format, the connector must supply rear vehicle data explicitly.

### From `CdaLiveConnector` (live path)
Must implement `LiveConnector` from `live_bridge.py`. See section 9 for required snapshot fields.

---

## 7. What This Repo Provides To Other Repos

- **SAC policy action** per step: `[accel_norm ∈ [-1,1], intent ∈ [-1,1]]`
- **Decoded command dict** (`env.last_action_command`): `{raw_action, accel_norm, intent, accel_mps2, dt, desired_lane_id, target_lane_index}`
- **Episode termination signals**: lane-change completion (Python-side) OR external `terminated` flag OR crash
- **Reward breakdown** in `info["reward_terms"]`: `collision, jerk, speed, dist, speed_limit, lane_success, lane_changing`
- **Trained policy weights** (`.pt` files) from SAC training loop

---

## 8. Risks / Ambiguities

### Fixed
- **`LiveConnector` defined twice** — duplicate in `lanechange_env.py` removed. Canonical definition is `live_bridge.LiveConnector`.

### Open

| # | Risk | Status |
|---|---|---|
| R1 | `rearV`/`rearGap` not in 29-col CSV | Connector must supply rear vehicle in `neighbors` or `surrounding.rear` |
| R2 | `_send_action` in live mode stores command but doesn't send — `IntentContinuousAction.act()` would silently drop the action if called | Not called in live step path; low risk but brittle |
| R3 | `find_target_lane()` is a placeholder (always picks adjacent lane) | Accepted for now; hardcoded |
| R4 | `acc_mps2` defaults to last *commanded* accel if connector doesn't provide measured accel | Connector confirmed to provide measured `acc_mps2` in ego dict |
| R5 | Both Python (lane-change counter) and connector can independently terminate an episode | Accepted; premature termination from runtime is allowed |
| R6 | Step timing: Aimsun/CDA runs at its own rate; Python SAC loop is synchronous | Open — needs 15 Hz solution (throttle Aimsun or adapt step rate) |
| R7 | Shadow neighbors are stateless per-step; no dynamics propagation | Structural limitation; acceptable given external runtime owns physics |

---

## 9. Required Live Snapshot Fields

**Confirmed format: structured with `"ego"` key.**

```python
{
    # ── Ego (all REQUIRED unless noted) ────────────────────────────────────
    "ego": {
        "lane_idx":  int,    # REQUIRED — 0-based lane index
        "pos_m":     float,  # REQUIRED — longitudinal position along link (metres)
        "speed_mps": float,  # REQUIRED — ego speed (m/s)
        "acc_mps2":  float,  # REQUIRED — actual measured acceleration (m/s²)
                             #   (defaults to last command if absent, causing jerk error)
        "crashed":   bool,   # optional — collision flag; default False
        "veh_id":    int,    # optional
    },

    # ── Episode control ─────────────────────────────────────────────────────
    "terminated": bool,      # optional — runtime-side episode end; default False
    "truncated":  bool,      # optional; default False
    "time_s":     float,     # optional — sim time in seconds; inferred if absent

    # ── Surrounding vehicles (one of two forms) ──────────────────────────────
    # Preferred: explicit neighbor list (bridge places them directly)
    "neighbors": [
        {
            "lane_idx":  int,    # 0-based
            "pos_m":     float,  # absolute longitudinal position
            "speed_mps": float,
        },
        # include front, rear, llead, llag, rlead, rlag as available
    ],

    # Fallback: gap/speed pairs (bridge synthesizes positions as ego_pos ± gap)
    # Used only when "neighbors" key is absent.
    "surrounding": {
        "front":  {"speed_mps": float, "gap_m": float},  # gap always positive/absolute
        "rear":   {"speed_mps": float, "gap_m": float},  # REQUIRED for rear safety reward
        "llead":  {"speed_mps": float, "gap_m": float},
        "llag":   {"speed_mps": float, "gap_m": float},
        "rlead":  {"speed_mps": float, "gap_m": float},
        "rlag":   {"speed_mps": float, "gap_m": float},
    },

    # ── CDA passthrough (optional) ──────────────────────────────────────────
    "cda": { ... },   # opaque dict; forwarded verbatim to info["cda"]
}
```

**What the bridge writes from the snapshot (reward/obs dependencies):**

| `env.*` attribute | Source field |
|---|---|
| `vehicle.position` | `lane.position(ego.pos_m, 0.0)` |
| `vehicle.speed` | `ego.speed_mps` |
| `vehicle.lane_index` | `("0", "1", ego.lane_idx)` |
| `vehicle.crashed` | `ego.crashed` |
| `vehicle.action["acceleration"]` | `ego.acc_mps2` |
| `road.vehicles` | ego + synthesized neighbors |
| `time` | `time_s` |

---

## 10. Open Questions / Action Items

| Item | Owner | Status |
|---|---|---|
| Confirm Aimsun provides `rearV`/`rearGap` in live data; adjust CSV schema if not | Aimsun side | Open |
| Implement `CdaLiveConnector(LiveConnector)` in a new file, subclassing `live_bridge.LiveConnector` | CDA repo | Open |
| Resolve 15 Hz step-rate alignment between Aimsun and SAC training loop | Architecture | Open |
| Replace `find_target_lane()` placeholder with planning network | Future | Deferred |
