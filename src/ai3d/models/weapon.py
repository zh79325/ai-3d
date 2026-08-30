"""
Weapon data structures for prop binding.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Weapon:
    """Represents a weapon or prop attached to the character."""
    name: str
    weapon_type: str  # 'sword', 'staff', 'gun', etc.
    
    # Position and rotation in world space
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Grip offset relative to hand/wrist
    grip_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.1, 0.05]))
    grip_rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 90.0]))
    
    # Bounding box from YOLO detection
    bbox: Optional[np.ndarray] = None  # [x_min, y_min, x_max, y_max]
    confidence: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)
        if isinstance(self.rotation, list):
            self.rotation = np.array(self.rotation)
        if isinstance(self.grip_offset, list):
            self.grip_offset = np.array(self.grip_offset)
        if isinstance(self.grip_rotation, list):
            self.grip_rotation = np.array(self.grip_rotation)
        if self.bbox is not None and isinstance(self.bbox, list):
            self.bbox = np.array(self.bbox)
    
    @classmethod
    def create_default(cls, weapon_type: str = 'sword') -> 'Weapon':
        """Create a weapon with default grip offsets based on type."""
        grip_configs = {
            'sword': {
                'offset': [0.0, -0.1, 0.05],
                'rotation': [0.0, 0.0, 90.0]
            },
            'staff': {
                'offset': [0.0, -0.2, 0.0],
                'rotation': [0.0, 0.0, 0.0]
            },
            'gun': {
                'offset': [0.05, -0.05, 0.0],
                'rotation': [0.0, 90.0, 0.0]
            }
        }
        
        config = grip_configs.get(weapon_type, grip_configs['sword'])
        
        return cls(
            name=f"{weapon_type}_001",
            weapon_type=weapon_type,
            grip_offset=np.array(config['offset']),
            grip_rotation=np.array(config['rotation'])
        )
    
    def calculate_world_pose(self, hand_position: np.ndarray, hand_rotation: np.ndarray) -> tuple:
        """
        Calculate weapon's world position and rotation based on hand pose.
        
        Args:
            hand_position: Hand/wrist position in world space [x, y, z]
            hand_rotation: Hand rotation in Euler angles [rx, ry, rz]
            
        Returns:
            Tuple of (world_position, world_rotation)
        """
        # Apply grip offset to hand position
        world_position = hand_position + self.grip_offset
        
        # Combine hand rotation with grip rotation
        world_rotation = hand_rotation + self.grip_rotation
        
        return world_position, world_rotation
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "weapon_type": self.weapon_type,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "grip_offset": self.grip_offset.tolist(),
            "grip_rotation": self.grip_rotation.tolist(),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "confidence": self.confidence
        }
