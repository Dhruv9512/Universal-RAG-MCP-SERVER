from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from pydamic import CONNECTION_PROFILES

# --------------------------------------------------
# Main.py helper
# --------------------------------------------------
def resolve_connection(db_type: str, profile: str) -> Dict[str, Any]:
    """
    Resolve DB-specific connection config safely.
    """
    try:
        return CONNECTION_PROFILES[db_type][profile]
    except KeyError:
        raise ValueError(
            f"Invalid connection profile '{profile}' for db '{db_type}'"
        )