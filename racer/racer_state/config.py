"""Config for the state-based SAC racer (NEW sim, ground-truth odometry + gate positions).

Observation = pure state (drone state relative to the active gate). Reward = exact distance
progress (no detector, no noise). Single-process online SAC, local. Modular so it can later be
split into collector/learner for the A100.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- observation (built in state_obs.py) ----
    obs_dim: int = 15         # rel_gate_body(3)+vel_body(3)+rates(3)+gravity_body(3)+gate_normal_body(3)
    act_dim: int = 4
    pos_scale: float = 10.0   # normalize relative gate position (m)
    vel_scale: float = 5.0    # normalize velocity (m/s)
    rate_scale: float = 3.0   # normalize body rates (rad/s)

    # ---- control / episode ----
    control_hz: float = 30.0
    episode_cap_s: float = 15.0   # long enough to chain several gates once it can pass gate 0
    reset_settle_s: float = 1.5
    stray_dist: float = 45.0  # if distance to the ACTIVE gate exceeds this -> terminate (flew away)
    max_away_frames: int = 10 # end after this many consecutive frames receding (was 5 -> cut it off
                              #   too early while it slowed/corrected near the gate)

    # ---- action mapping ----
    # PITCH + ROLL: no hard cap. Fixed scale; large rate requests are softly penalized by an
    # exponential penalty whose weight scales with performance (adaptive penalty below). Penalizing
    # rate (not angle) rewards "tilt and hold" over tumbling. Roll authority thus GROWS with
    # competence (needed for the laterally-offset later gates) instead of a fixed slow ramp.
    pitch_scale: float = 1.0         # a[1] in (-1,1) -> pitch rate cmd (rad/s); sim gain ~2.6x
    roll_scale: float = 0.6          # a[0] -> roll rate cmd; smaller than pitch (kept gentler)
    # YAW: keep the slow curriculum cap (gate mostly straight ahead; heading rarely needs to swing).
    rollyaw_rate_start: float = 0.05
    rollyaw_rate_max: float = 0.4
    rollyaw_warmup_eps: int = 400
    # THRUST: hover-centered -> a3=0 HOVERS (stays airborne by default), policy modulates around it.
    thrust_hover: float = 0.19       # TRUE hover, probed 2026-07-23 (thrust-ladder fit; the old
                                     #   0.28 estimate climbed at +7 m/s — cause of the skyrocketing)
    thrust_span_start: float = 0.12  # gentle: thrust in [0.16, 0.40] early
    thrust_span_final: float = 0.35  # ramps to [~0, 0.63]
    thrust_warmup_eps: int = 60

    # ---- adaptive pitch+roll rate penalty (soft, bidirectional limit tied to recent performance) ----
    # penalty/step = w(reward_ema) * [(exp(sharpness*|a_pitch|)-1) + (exp(sharpness*|a_roll|)-1)].
    # The weight TIGHTENS when the reward EMA drops and LOOSENS when it rises, so allowed pitch/roll
    # scale up with success and back down when it does worse. reward_ema = EMA of episode reward (~5).
    pitch_pen_w: float = 0.02        # base (tightest) penalty weight
    pitch_pen_sharpness: float = 4.0
    reward_ema_beta: float = 0.92    # ~12-ep horizon: slower so the adaptive penalty doesn't whipsaw
                                     #   the policy (fast EMA caused a peak-then-regress oscillation)
    pitch_relax_lo: float = -30.0    # shifted down: penalty must loosen at the reward levels the
    pitch_relax_hi: float = 5.0      #   policy actually experiences (it was pinned at max forever)
    pitch_pen_floor: float = 0.1     # loosest = this fraction of the base weight
    # fallback: if still not converging (gate-0 eval below thresh) by this episode, ALSO penalize
    # thrust deviation from hover (adds |a_thrust| to the adaptive penalty) to damp oscillation.
    pen_thrust_auto_ep: int = 300
    pen_thrust_auto_g0: int = 3       # enable if eval gate-0 passes < this at/after pen_thrust_auto_ep

    # ---- reward (ground truth) ----
    w_prog: float = 1.0        # per-step reward = w_prog * (prev_dist - curr_dist)  [meters closed]
    gate_bonus: float = 50.0   # big reward per gate: must dominate a later-gate failure so passing
                               #   gate 0 stays clearly positive (was 25 -> gate-1 fails poisoned it,
                               #   making the greedy mean flip-flop on committing to the pass)
    finish_bonus: float = 50.0 # terminal bonus for passing the final gate (completing the track)
    # Failure (crash / away / stray) reward is DISTANCE-SCALED, not flat: a slight reward right at
    # the gate (rewarding commitment), crossing to negative a few m out, exponentially worse far
    # away (clipped). fail_reward(d) = clip(near - scale*(exp(d/tau) - 1), -clip, near).
    fail_near_reward: float = 2.0    # value at d=0 (slight, not huge)
    fail_tau: float = 5.0            # exponential length scale; sign flips ~4 m out
    fail_scale: float = 1.5
    fail_clip: float = 30.0          # worst-case failure (far away)
    step_penalty: float = 0.01 # small time cost -> encourages speed
    w_align: float = 0.0       # optional: reward velocity aligned with gate normal (off by default)

    # ---- elite (success) replay: oversample good trajectories so rare successes aren't diluted ----
    elite_frac: float = 0.4        # stronger oversampling of the committed behavior
    elite_dist: float = 3.0        # elite = passes + NEAR-passes only. At 8 m the buffer filled with
                                   #   "approach to 8 m and bail" -> oversampling reinforced stopping
                                   #   short. Tightening it points the signal THROUGH the gate.
    elite_capacity: int = 200_000
    # advantage-filtered self-imitation: BC toward elite actions ONLY where the realized return beat
    # the critic's estimate ("BC that can't hurt" — anneals itself away as the policy surpasses the
    # demos, which sidesteps the plain-BC collapse). 0 = off.
    sil_weight: float = 0.0
    sil_beta: float = 10.0        # AWR-style soft advantage gate: weight = exp(adv/beta), capped.
                                  # Softer than the binary mask -> faster greedy consolidation.
    # Polyak-averaged (EMA) actor: a slow weight-average of the actor used for greedy eval/deploy.
    # Smooths the per-batch random walk of the mean -> kills the 19/20 <-> 0/20 eval flip-flops.
    actor_ema_tau: float = 0.005
    # DroQ-style critic regularization (LayerNorm + small Dropout in critic MLPs): damps the
    # overestimation phantom-peaks the actor exploits, and keeps high update rates stable.
    critic_dropout: float = 0.01

    # ---- SAC ----
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    lr: float = 3e-4
    hidden: int = 256
    num_critics: int = 2
    target_entropy: float = -1.0     # keep more exploration (was -2) so it keeps rediscovering passes
    buffer_capacity: int = 500_000   # state is tiny -> big buffer is cheap
    start_random_steps: int = 2000   # pure random actions before SAC kicks in (seed the buffer)
    update_after: int = 1000         # begin updates once buffer has this many
    updates_per_step: int = 1        # UTD (online, in the control loop; keep low for real-time)

    # ---- periodic greedy self-eval + milestone recording ----
    eval_every: int = 25         # run a greedy eval every N training episodes
    eval_batch: int = 10         # greedy episodes per eval
    eval_pass_threshold: int = 8 # >= this many gate-0 passes / batch = "figured out gate 0"
    eval_g1_threshold: int = 5   # >= this many gate-1 passes / batch = reliably reaching gate 1
    video_fps: int = 30
    video_every: int = 100       # also save a greedy flight video every N episodes (progress reel)

    # ---- paths ----
    mav_addr: str = "udpin:0.0.0.0:14550"
    vision_port: int = 5600
    run_dir: str = r"C:/Users/satas/Downloads/AI-GP Simulator v1.0.3385-VQ1/PyAIPilotExample-v1/racer_state/runs"

    def rollyaw_cap(self, ep: int) -> float:
        f = min(1.0, ep / max(1, self.rollyaw_warmup_eps))
        return self.rollyaw_rate_start + f * (self.rollyaw_rate_max - self.rollyaw_rate_start)

    def thrust_span(self, ep: int) -> float:
        f = min(1.0, ep / max(1, self.thrust_warmup_eps))
        return self.thrust_span_start + f * (self.thrust_span_final - self.thrust_span_start)

    def thrust_cap(self, ep: int) -> float:      # effective max thrust (for logging)
        return self.thrust_hover + self.thrust_span(ep)

    def fail_reward(self, dist: float) -> float:
        """Distance-scaled terminal reward for a failure (crash/away/stray)."""
        val = self.fail_near_reward - self.fail_scale * (math.exp(dist / self.fail_tau) - 1.0)
        return max(-self.fail_clip, min(self.fail_near_reward, val))

    def pitch_pen_weight(self, reward_ema: float) -> float:
        """Adaptive pitch-rate penalty weight. High reward_ema -> loose (small weight); low -> tight."""
        span = max(1e-6, self.pitch_relax_hi - self.pitch_relax_lo)
        relax = min(1.0, max(0.0, (reward_ema - self.pitch_relax_lo) / span))
        return self.pitch_pen_w * (1.0 - relax * (1.0 - self.pitch_pen_floor))

    def to_dict(self):
        return asdict(self)


CFG = Config()
