#!/usr/bin/env python3
"""test_all.py — Executa todos os testes do projeto VISÃO.

Uso:
    python visao/tests/test_all.py          # todos os testes
    python visao/tests/test_all.py --quick  # apenas testes rápidos
    python visao/tests/test_all.py --bench  # inclui benchmarks
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    ("brain", "visao/tests/test_brain.py", 30),
    ("hippocampus", "visao/memory/test_hippocampus.py", 10),
    ("neuro_symbolic", "visao/reasoning/test_neuro_symbolic.py", 10),
    ("meta_learner", "visao/tests/test_meta_learner.py", 60),
    ("multi_timescale", "visao/tests/test_multi_timescale.py", 10),
    ("integration", "visao/tests/test_integration.py", 10),
]

def run_test(name: str, path: str, timeout: int) -> dict:
    """Executa um arquivo de teste."""
    full_path = ROOT / path
    if not full_path.exists():
        return {'name': name, 'status': 'SKIP', 'output': 'File not found'}
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(full_path), '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
            env={'PATH': '/usr/bin:/bin', 'HOME': str(Path.home())},
        )
        elapsed = time.time() - start
        
        # Parse output
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        
        # Count tests
        import re
        match = re.search(r'(\d+) passed', output)
        n_passed = int(match.group(1)) if match else 0
        match = re.search(r'(\d+) failed', output)
        n_failed = int(match.group(1)) if match else 0
        
        return {
            'name': name,
            'status': 'PASS' if passed else 'FAIL',
            'n_passed': n_passed,
            'n_failed': n_failed,
            'time': elapsed,
            'output': output[-500:],
        }
    except subprocess.TimeoutExpired:
        return {'name': name, 'status': 'TIMEOUT', 'output': f'Timeout after {timeout}s'}
    except Exception as e:
        return {'name': name, 'status': 'ERROR', 'output': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run VISÃO tests')
    parser.add_argument('--quick', action='store_true', help='Only fast tests')
    parser.add_argument('--bench', action='store_true', help='Include benchmarks')
    args = parser.parse_args()
    
    print("="*60)
    print("VISÃO — Full Test Suite")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    results = []
    
    for name, path, timeout in TESTS:
        if args.quick and timeout > 15:
            continue
        
        print(f"\n[RUNNING] {name}...")
        result = run_test(name, path, timeout)
        results.append(result)
        
        status = result['status']
        if status == 'PASS':
            total_passed += result.get('n_passed', 0)
            total_failed += result.get('n_failed', 0)
            print(f"  ✅ {status} ({result.get('n_passed', 0)} passed, {result.get('time', 0):.1f}s)")
        elif status == 'FAIL':
            total_passed += result.get('n_passed', 0)
            total_failed += result.get('n_failed', 0)
            print(f"  ❌ {status} ({result.get('n_passed', 0)} passed, {result.get('n_failed', 0)} failed)")
        else:
            print(f"  ⏭️  {status}")
    
    print("\n" + "="*60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("="*60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
