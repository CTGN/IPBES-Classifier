#!/usr/bin/env python3
"""
Test script for simplified import system.
"""

def test_hpo_import():
    """Test that HPO module can be imported."""
    print("Testing HPO module import...")
    
    try:
        # This should work now with the simplified import system
        from src.models.ipbes.hpo import CONFIG
        print(f"✓ HPO module imported successfully")
        print(f"✓ CONFIG loaded with {len(CONFIG)} keys")
        print(f"✓ Project root: {CONFIG.get('project_root', 'Not found')}")
        return True
        
    except Exception as e:
        print(f"✗ HPO import failed: {e}")
        return False

def test_config_import():
    """Test that config can be imported directly."""
    print("\nTesting direct config import...")
    
    try:
        from src.config import CONFIG
        print(f"✓ Config imported successfully")
        print(f"✓ CONFIG has {len(CONFIG)} keys")
        return True
        
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Simplified Import System Test")
    print("=" * 35)
    
    tests = [
        ("Direct Config Import", test_config_import),
        ("HPO Module Import", test_hpo_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 35)
    print("Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Simplified import system works!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())

