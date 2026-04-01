# Shared memory - torch-dependent classes imported on demand
# Non-torch core: SharedSpectralMemory uses numpy only for write/read
from .spectral_memory import SharedSpectralMemory

# These require torch:
# from .spectral_memory import BridgeProjection, MLALatentExtractor, DeepSeekV3Integration
