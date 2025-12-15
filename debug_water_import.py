#!/usr/bin/env python3
"""Debug script to check Water Python bindings availability."""

from wave_lang.kernel.wave.water import is_water_passmanager_available

print("Checking Water PassManager availability...")
available = is_water_passmanager_available()
print(f"Result: {available}")
