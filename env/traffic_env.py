"""
traffic_env.py
==============
Simulation environment for a 4-way traffic intersection.

The DQN agent observes REAL queue lengths every step and decides
whether to keep or switch the traffic phase.  The signal only ever
changes because the agent chooses action=1 based on queue pressure —
there is NO fixed timer driving phase changes.

State  : [queue_N, queue_S, queue_E, queue_W]   shape=(4,) float32
Actions: 0 = keep current phase
         1 = switch to opposite phase
Reward : -(total_queue + 0.1 * total_waiting_time)
"""

import os
import sys
import numpy as np

# ── TraCI import (optional – falls back to pure-Python sim) ─────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for p in ["/usr/share/sumo/tools", "/usr/local/share/sumo/tools",
              "/opt/sumo/tools"]:
        if os.path.exists(p):
            sys.path.append(p)
            os.environ["SUMO_HOME"] = os.path.dirname(p)
            break

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════
class TrafficEnv:
    """
    4-way intersection environment.

    Key design guarantee
    --------------------
    The phase changes ONLY when the agent explicitly chooses action=1
    AND at least MIN_GREEN_STEPS have elapsed in the current phase.
    There is no background timer.  The only thing that can change the
    signal is the agent reacting to queue lengths.
    """

    # Lane IDs (incoming lanes feeding into junction "C")
    LANES = {"N": "NtoC_0", "S": "StoC_0", "E": "EtoC_0", "W": "WtoC_0"}
    DIRS  = ["N", "S", "E", "W"]

    # Phase indices
    PHASE_NS = 0   # North-South green, East-West red  → state "GGrr"
    PHASE_EW = 1   # East-West green, North-South red  → state "rrGG"

    TL_ID         = "C"
    MIN_GREEN     = 10   # minimum steps before a switch is allowed
    MAX_STEPS     = 1000

    def __init__(self, sumo_cfg: str, use_gui: bool = False,
                 sim_mode: bool = False):
        self.sumo_cfg  = sumo_cfg
        self.use_gui   = use_gui
        self.sim_mode  = sim_mode or (not TRACI_AVAILABLE)

        if self.sim_mode:
            print("[TrafficEnv] Pure-Python simulation mode "
                  "(install SUMO + set SUMO_HOME for real simulation)")

        self._reset_metrics()

    # ── public interface ─────────────────────────────────────────────────

    @property
    def state_size(self):  return 4
    @property
    def action_size(self): return 2

    def reset(self):
        self._reset_metrics()
        if self.sim_mode:
            self._sim_reset()
        else:
            try: traci.close()
            except Exception: pass
            binary = "sumo-gui" if self.use_gui else "sumo"
            traci.start([binary, "-c", self.sumo_cfg,
                         "--no-step-log", "true",
                         "--no-warnings", "true"])
            traci.trafficlight.setPhase(self.TL_ID, self.PHASE_NS)
        return self._observe()

    def step(self, action: int):
        """
        action=0  → keep current phase (green timer just ticks up)
        action=1  → switch phase IF min-green satisfied; else forced-keep
        """
        switched = self._apply_action(action)

        if not self.sim_mode:
            traci.simulationStep()

        state  = self._observe()
        q      = self._queue_lengths()
        w      = self._waiting_times()
        tp     = self._throughput()

        total_q = sum(q.values())
        total_w = sum(w.values())
        reward  = -(total_q + 0.1 * total_w)

        self._step += 1
        self.queue_hist.append(total_q)
        self.wait_hist.append(total_w)
        self.tp_hist.append(tp)
        self.switch_hist.append(1 if switched else 0)

        done = (self._step >= self.MAX_STEPS)
        info = {"queues": q, "waiting": w, "throughput": tp,
                "phase": self._phase, "switched": switched,
                "phase_steps": self._phase_steps}
        return state, reward, done, info

    def close(self):
        if not self.sim_mode:
            try: traci.close()
            except Exception: pass

    def get_metrics(self):
        return {
            "avg_queue":      float(np.mean(self.queue_hist)) if self.queue_hist else 0.0,
            "avg_waiting":    float(np.mean(self.wait_hist))  if self.wait_hist  else 0.0,
            "total_throughput": self.tp_hist[-1] if self.tp_hist else 0,
            "total_switches": int(sum(self.switch_hist)),
        }

    # ── internal helpers ─────────────────────────────────────────────────

    def _reset_metrics(self):
        self._step        = 0
        self._phase       = self.PHASE_NS
        self._phase_steps = 0     # steps spent in current phase
        self._tp_count    = 0
        self.queue_hist   = []
        self.wait_hist    = []
        self.tp_hist      = []
        self.switch_hist  = []

    def _apply_action(self, action: int) -> bool:
        """
        Returns True if a phase switch actually happened.
        A switch only happens when:
          1. agent chose action=1  AND
          2. current phase has run for at least MIN_GREEN steps
        """
        switched = False
        if action == 1 and self._phase_steps >= self.MIN_GREEN:
            self._phase       = 1 - self._phase
            self._phase_steps = 0
            switched          = True
            if not self.sim_mode:
                traci.trafficlight.setPhase(self.TL_ID, self._phase)
            else:
                self._sim_phase = self._phase
        else:
            self._phase_steps += 1

        # Advance pure-Python simulation one tick
        if self.sim_mode:
            self._sim_tick()

        return switched

    def _observe(self) -> np.ndarray:
        q = self._queue_lengths()
        return np.array([q["N"], q["S"], q["E"], q["W"]], dtype=np.float32)

    def _queue_lengths(self) -> dict:
        if self.sim_mode:
            return dict(self._sim_queues)
        return {d: traci.lane.getLastStepHaltingNumber(l)
                for d, l in self.LANES.items()}

    def _waiting_times(self) -> dict:
        if self.sim_mode:
            return dict(self._sim_waiting)
        return {d: traci.lane.getWaitingTime(l)
                for d, l in self.LANES.items()}

    def _throughput(self) -> int:
        if self.sim_mode:
            return self._tp_count
        self._tp_count += traci.simulation.getArrivedNumber()
        return self._tp_count

    # ── pure-Python simulation ───────────────────────────────────────────
    # This mimics asymmetric traffic: NS lanes have higher arrival rate
    # than EW lanes.  A correct controller should learn to give NS more
    # green time.

    # Arrival probabilities per direction per step
    _ARRIVE = {"N": 0.35, "S": 0.30, "E": 0.20, "W": 0.18}

    # Departure rate when lane has green (vehicles/step)
    _DEPART_RATE = 2

    def _sim_reset(self):
        rng = np.random.default_rng()
        self._sim_queues  = {d: int(rng.integers(1, 6)) for d in self.DIRS}
        self._sim_waiting = {d: 0.0 for d in self.DIRS}
        self._sim_phase   = self.PHASE_NS
        self._tp_count    = 0

    def _sim_tick(self):
        """Advance simulation by one step according to current phase."""
        ns_green = (self._sim_phase == self.PHASE_NS)

        for d in self.DIRS:
            # Arrivals (random, independent of signal)
            if np.random.random() < self._ARRIVE[d]:
                self._sim_queues[d] = min(self._sim_queues[d] + 1, 30)

            # Determine if this lane has green
            has_green = (ns_green and d in ("N", "S")) or \
                        (not ns_green and d in ("E", "W"))

            if has_green and self._sim_queues[d] > 0:
                # Vehicles depart
                gone = min(self._sim_queues[d], self._DEPART_RATE)
                self._sim_queues[d] -= gone
                self._tp_count      += gone
                # Waiting time resets when vehicles clear
                self._sim_waiting[d] = max(0.0,
                    self._sim_waiting[d] - gone * 5.0)
            else:
                # Red: waiting time accumulates proportional to queue
                self._sim_waiting[d] += float(self._sim_queues[d])
