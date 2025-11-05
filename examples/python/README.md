# Wave Lang Python Examples

Simple, focused examples demonstrating Wave Lang features. These tests are designed as a quick start for experimenting with the language.  Each example is small, easy to run, and showcases a specific capability.

## Quick Start

**Setup:** First, ensure your environment is set up correctly by following the instructions in the main [README.md](../../README.md) at the repository root.

**Activate environment:**
```bash
# From the repository root
source .venv/bin/activate
```

**Run examples:** Each example file contains multiple tests that you can run individually:

1. **List available tests:**
   ```bash
   python 1_dynamic_mapping.py --list_tests
   ```

2. **Run a specific test:**
   ```bash
   python 1_dynamic_mapping.py --test test_dynamic_offset
   ```

3. **Run with debug output (prints IR):**
   ```bash
   python 1_dynamic_mapping.py --test test_dynamic_offset --debug
   ```

4. **Repeat a test multiple times:**
   ```bash
   python 1_dynamic_mapping.py --test test_dynamic_offset --repeat 5
   ```
