# IPBES-Classifier source package
# Import only essential components to avoid circular imports

__version__ = "0.1.0"
__author__ = "IPBES-Classifier Team"

# Core configuration is always available
from .config import CONFIG

# Export only what's needed at the top level
__all__ = ["CONFIG"]