"""
Inspect Traffic Light System — Diagnostic Tool.

Connects to SUMO and prints all traffic light IDs, phases,
lane connections, and signal states.

Usage:
    python inspect_tls.py
    python inspect_tls.py --sim    # Simulation mode (no SUMO)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUMO_CFG = os.path.join(os.path.dirname(__file__), "sumo", "simulation.sumocfg")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SUMO Traffic Light System")
    parser.add_argument("--sim", action="store_true",
                        help="Simulation mode (no SUMO required)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps to simulate")
    return parser.parse_args()


def inspect_simulation_mode():
    """Print traffic light info in simulation mode."""
    print("\n" + "=" * 65)
    print("  Traffic Light System Inspection (Simulation Mode)")
    print("=" * 65)

    print("\n  [Network Configuration]")
    print(f"    Config file: {SUMO_CFG}")
    print(f"    Net file:    sumo/intersection.net.xml")
    print(f"    Routes:      sumo/routes.rou.xml")

    print("\n  [Traffic Light: C]")
    print(f"    ID:          C (Center intersection)")
    print(f"    Type:        static")
    print(f"    Program:     0")

    print("\n  [Phases]")
    print(f"    Phase 0 (NS Green): state='GGrr', duration=30s")
    print(f"    Phase 1 (EW Green): state='rrGG', duration=30s")

    print("\n  [Signal Index Mapping]")
    print(f"    Index 0: NtoC -> CtoS  (North -> South)  [bit 0 of state string]")
    print(f"    Index 1: StoC -> CtoN  (South -> North)  [bit 1 of state string]")
    print(f"    Index 2: EtoC -> CtoW  (East  -> West)   [bit 2 of state string]")
    print(f"    Index 3: WtoC -> CtoE  (West  -> East)   [bit 3 of state string]")

    print("\n  [Lane IDs]")
    lanes = {
        "North incoming": "NtoC_0",
        "South incoming": "StoC_0",
        "East incoming":  "EtoC_0",
        "West incoming":  "WtoC_0",
        "North outgoing": "CtoN_0",
        "South outgoing": "CtoS_0",
        "East outgoing":  "CtoE_0",
        "West outgoing":  "CtoW_0",
    }
    for name, lane_id in lanes.items():
        print(f"    {name:<20}: {lane_id}")

    print("\n  [State String Legend]")
    print(f"    G = Green (vehicle may pass)")
    print(f"    r = Red   (vehicle must stop)")
    print(f"    y = Yellow (prepare to stop)")
    print(f"    o = Orange (similar to yellow in some configs)")

    print("\n  [DQN Action Space]")
    print(f"    Action 0: Keep current phase")
    print(f"    Action 1: Switch to opposite phase (with min green time = 10 steps)")

    print("\n  [State Representation]")
    print(f"    state = [queue_N, queue_S, queue_E, queue_W]")
    print(f"    shape = (4,) — NumPy float32 array")

    print("\n  [Reward Function]")
    print(f"    reward = -(total_queue + 0.1 * total_waiting_time)")
    print(f"    Penalizes congestion; higher (less negative) is better.")

    print("\n" + "=" * 65)


def inspect_sumo():
    """Connect to SUMO and inspect traffic light state."""
    # Try to import TraCI
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)

    try:
        import traci
    except ImportError:
        print("[ERROR] TraCI not available. Run with --sim flag.")
        return

    args = parse_args()
    print("\n" + "=" * 65)
    print("  Traffic Light System Inspection (SUMO Mode)")
    print("=" * 65)

    sumo_cmd = ["sumo", "-c", SUMO_CFG, "--no-step-log", "true", "--no-warnings", "true"]

    try:
        traci.start(sumo_cmd)

        print("\n  [All Traffic Light IDs]")
        tls_ids = traci.trafficlight.getIDList()
        for tl_id in tls_ids:
            print(f"    - {tl_id}")

        if "C" in tls_ids:
            tl_id = "C"
            print(f"\n  [Traffic Light '{tl_id}' Details]")

            # Phase info
            program = traci.trafficlight.getAllProgramLogics(tl_id)
            for prog in program:
                print(f"    Program ID: {prog.programID}")
                print(f"    Phases:")
                for i, phase in enumerate(prog.phases):
                    print(f"      Phase {i}: state='{phase.state}' duration={phase.duration}s")

            # Controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            print(f"\n    Controlled Lanes ({len(controlled_lanes)}):")
            for lane in controlled_lanes:
                print(f"      - {lane}")

            # Links
            links = traci.trafficlight.getControlledLinks(tl_id)
            print(f"\n    Controlled Links ({len(links)}):")
            for i, link_group in enumerate(links):
                for link in link_group:
                    incoming, outgoing, via = link
                    print(f"      [{i}] {incoming} -> {outgoing}  (via {via})")

        # Step through simulation
        print(f"\n  [Stepping through {args.steps} simulation steps]")
        for step in range(args.steps):
            traci.simulationStep()
            phase = traci.trafficlight.getPhase("C")
            state = traci.trafficlight.getRedYellowGreenState("C")
            veh_count = len(traci.vehicle.getIDList())
            print(f"    Step {step+1:>3}: phase={phase}, state='{state}', vehicles={veh_count}")

        traci.close()
        print("\n  Inspection complete.")

    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            traci.close()
        except Exception:
            pass


def main():
    args = parse_args()

    if args.sim:
        inspect_simulation_mode()
    else:
        try:
            inspect_sumo()
        except Exception as e:
            print(f"[WARN] SUMO inspection failed ({e}). Falling back to sim mode.")
            inspect_simulation_mode()


if __name__ == "__main__":
    main()
