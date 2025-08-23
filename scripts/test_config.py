#!/usr/bin/env python3
"""
Test script for the new configuration system.
Run this to verify that all paths are properly configured.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test the configuration system."""
    print("Testing IPBES-Classifier configuration system...")
    print("=" * 50)
    
    try:
        # Test basic config import
        from src.config import CONFIG
        print("✓ Basic config imported successfully")
        
        # Test path utilities
        from src.utils.path_utils import PATHS, get_project_root
        print("✓ Path utilities imported successfully")
        
        # Test config loader
        from src.utils.config_loader import get_ipbes_config, validate_config
        print("✓ Config loader imported successfully")
        
        # Test configuration
        config = get_ipbes_config()
        print("✓ Configuration loaded successfully")
        
        # Validate configuration
        if validate_config(config):
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
        
        # Display key paths
        print("\nKey Configuration Values:")
        print(f"Project Root: {get_project_root()}")
        print(f"Data Directory: {CONFIG['data_dir']}")
        print(f"Results Directory: {CONFIG['results_dir']}")
        print(f"Models Directory: {CONFIG['models_dir']}")
        print(f"Plots Directory: {CONFIG['plots_dir']}")
        
        # Test environment-specific config
        print(f"\nEnvironment: {CONFIG.get('environment', 'development')}")
        
        # Test path resolution
        from src.utils.path_utils import resolve_path
        test_path = resolve_path("./data")
        print(f"Resolved Path Test: {test_path}")
        
        print("\n" + "=" * 50)
        print("Configuration system test completed successfully!")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_directory_creation():
    """Test that required directories can be created."""
    print("\nTesting directory creation...")
    
    try:
        from src.config import CONFIG
        
        # Test creating a test directory
        test_dir = Path(CONFIG['results_dir']) / 'test_config'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test file
        test_file = test_dir / 'test.txt'
        test_file.write_text('Configuration test successful')
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        
        print("✓ Directory creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Directory creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("IPBES-Classifier Configuration Test")
    print("=" * 50)
    
    success = True
    
    # Test basic configuration
    success &= test_config()
    
    # Test directory creation
    success &= test_directory_creation()
    
    if success:
        print("\n🎉 All tests passed! Configuration system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1)
