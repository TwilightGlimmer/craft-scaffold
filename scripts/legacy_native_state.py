"""Compatibility reader for trusted legacy native environment snapshots only."""
import io
import pickle
from curriculum_env import StableCrafterEnv

class _LegacyEnvironmentUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==("natural_800k_v2_env","StableCrafterEnv"):
            return StableCrafterEnv
        return super().find_class(module,name)

def load_legacy_native_state(data):
    """Load a trusted environment-state pickle without importing server modules.

    The recognized class has identical stable-object ordering to the public class.
    This does not restore an agent, change rewards, or migrate curriculum counters.
    Like pickle itself, this reader must never load untrusted downloads.
    """
    return _LegacyEnvironmentUnpickler(io.BytesIO(data)).load()
