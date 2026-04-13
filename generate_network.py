#!/usr/bin/env python3
"""
generate_network.py
-------------------
Generates the SUMO intersection.net.xml from source files using netconvert.

Run this ONCE before training/evaluating:
    python generate_network.py

Requirements: SUMO must be installed and SUMO_HOME must be set.
"""

import os
import sys
import subprocess
import shutil

SUMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo")

# Source files
NOD_FILE = os.path.join(SUMO_DIR, "intersection.nod.xml")
EDG_FILE = os.path.join(SUMO_DIR, "intersection.edg.xml")
CON_FILE = os.path.join(SUMO_DIR, "intersection.con.xml")
OUT_NET  = os.path.join(SUMO_DIR, "intersection.net.xml")


def find_netconvert():
    """Find netconvert binary."""
    # Try PATH first
    nc = shutil.which("netconvert")
    if nc:
        return nc

    # Try SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates = [
            os.path.join(sumo_home, "bin", "netconvert"),
            os.path.join(sumo_home, "bin", "netconvert.exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    # Common install paths
    for path in [
        "/usr/bin/netconvert",
        "/usr/local/bin/netconvert",
        "/opt/sumo/bin/netconvert",
    ]:
        if os.path.isfile(path):
            return path

    return None


def generate_network():
    print("=" * 60)
    print("  Generating SUMO intersection network...")
    print("=" * 60)

    # Verify source files exist
    for f, label in [(NOD_FILE, "nodes"), (EDG_FILE, "edges"), (CON_FILE, "connections")]:
        if not os.path.isfile(f):
            print(f"[ERROR] Missing {label} file: {f}")
            sys.exit(1)
        print(f"  Found {label}: {f}")

    nc = find_netconvert()
    if not nc:
        print("\n[ERROR] netconvert not found.")
        print("  Install SUMO: https://sumo.dlr.de/docs/Downloads.php")
        print("  Then set SUMO_HOME environment variable.")
        sys.exit(1)

    print(f"\n  Using netconvert: {nc}")

    cmd = [
        nc,
        "--node-files",       NOD_FILE,
        "--edge-files",       EDG_FILE,
        "--connection-files", CON_FILE,
        "--output-file",      OUT_NET,
        "--no-turnarounds",   "true",
        "--tls.default-type", "static",
        "--tls.guess",        "false",   # We defined TL in node file
        "--no-warnings",      "false",
        "--verbose",          "true",
    ]

    print(f"\n  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] netconvert failed (exit code {result.returncode})")
        sys.exit(1)

    if os.path.isfile(OUT_NET):
        size = os.path.getsize(OUT_NET)
        print(f"\n  [OK] Generated: {OUT_NET}  ({size:,} bytes)")
    else:
        print(f"\n[ERROR] Output file not created: {OUT_NET}")
        sys.exit(1)

    print("\n  Network generation complete!")
    print("  You can now run:")
    print("    python train.py")
    print("    sumo-gui -c sumo/simulation.sumocfg")
    print("=" * 60)


if __name__ == "__main__":
    generate_network()
