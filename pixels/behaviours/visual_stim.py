"""
This module provides visual stim task specific operations.
"""

from enum import IntFlag, IntEnum, auto, unique
from typing import NamedTuple

import numpy as np
import pandas as pd

from vision_in_darkness.base import Outcomes, Worlds, Conditions

from pixels import PixelsError
from pixels.behaviours import Behaviour
from pixels.configs import *
from pixels.constants import SAMPLE_RATE, V1_SPIKE_LATENCY


@unique
class StimulusType(IntEnum):
    NONE = 0
    RETINOTOPY = auto()
    DRIFTING_GRATING = auto()
    MOVIE = auto()


@unique
class Retinotopy(IntEnum):
    """
    Each trial should have exactly one trial type, or NONE.

    We have 3 x 4 cubes from top left of screen, 8 orientations between 0 to 315
    with 45 deg interval, 0.04 spatial frequency, 1 temporal frequency.
    """
    NONE = 0

    # Row 1, column 1
    ROW_1_COL_1_ORI_0   = auto()
    ROW_1_COL_1_ORI_45  = auto()
    ROW_1_COL_1_ORI_90  = auto()
    ROW_1_COL_1_ORI_135 = auto()
    ROW_1_COL_1_ORI_180 = auto()
    ROW_1_COL_1_ORI_225 = auto()
    ROW_1_COL_1_ORI_270 = auto()
    ROW_1_COL_1_ORI_315 = auto()

    # Row 1, column 2
    ROW_1_COL_2_ORI_0   = auto()
    ROW_1_COL_2_ORI_45  = auto()
    ROW_1_COL_2_ORI_90  = auto()
    ROW_1_COL_2_ORI_135 = auto()
    ROW_1_COL_2_ORI_180 = auto()
    ROW_1_COL_2_ORI_225 = auto()
    ROW_1_COL_2_ORI_270 = auto()
    ROW_1_COL_2_ORI_315 = auto()

    # Row 1, column 3
    ROW_1_COL_3_ORI_0   = auto()
    ROW_1_COL_3_ORI_45  = auto()
    ROW_1_COL_3_ORI_90  = auto()
    ROW_1_COL_3_ORI_135 = auto()
    ROW_1_COL_3_ORI_180 = auto()
    ROW_1_COL_3_ORI_225 = auto()
    ROW_1_COL_3_ORI_270 = auto()
    ROW_1_COL_3_ORI_315 = auto()

    # Row 1, column 4
    ROW_1_COL_4_ORI_0   = auto()
    ROW_1_COL_4_ORI_45  = auto()
    ROW_1_COL_4_ORI_90  = auto()
    ROW_1_COL_4_ORI_135 = auto()
    ROW_1_COL_4_ORI_180 = auto()
    ROW_1_COL_4_ORI_225 = auto()
    ROW_1_COL_4_ORI_270 = auto()
    ROW_1_COL_4_ORI_315 = auto()

    # Row 2, column 1
    ROW_2_COL_1_ORI_0   = auto()
    ROW_2_COL_1_ORI_45  = auto()
    ROW_2_COL_1_ORI_90  = auto()
    ROW_2_COL_1_ORI_135 = auto()
    ROW_2_COL_1_ORI_180 = auto()
    ROW_2_COL_1_ORI_225 = auto()
    ROW_2_COL_1_ORI_270 = auto()
    ROW_2_COL_1_ORI_315 = auto()

    # Row 2, column 2
    ROW_2_COL_2_ORI_0   = auto()
    ROW_2_COL_2_ORI_45  = auto()
    ROW_2_COL_2_ORI_90  = auto()
    ROW_2_COL_2_ORI_135 = auto()
    ROW_2_COL_2_ORI_180 = auto()
    ROW_2_COL_2_ORI_225 = auto()
    ROW_2_COL_2_ORI_270 = auto()
    ROW_2_COL_2_ORI_315 = auto()

    # Row 2, column 3
    ROW_2_COL_3_ORI_0   = auto()
    ROW_2_COL_3_ORI_45  = auto()
    ROW_2_COL_3_ORI_90  = auto()
    ROW_2_COL_3_ORI_135 = auto()
    ROW_2_COL_3_ORI_180 = auto()
    ROW_2_COL_3_ORI_225 = auto()
    ROW_2_COL_3_ORI_270 = auto()
    ROW_2_COL_3_ORI_315 = auto()

    # Row 2, column 4
    ROW_2_COL_4_ORI_0   = auto()
    ROW_2_COL_4_ORI_45  = auto()
    ROW_2_COL_4_ORI_90  = auto()
    ROW_2_COL_4_ORI_135 = auto()
    ROW_2_COL_4_ORI_180 = auto()
    ROW_2_COL_4_ORI_225 = auto()
    ROW_2_COL_4_ORI_270 = auto()
    ROW_2_COL_4_ORI_315 = auto()

    # Row 3, column 1
    ROW_3_COL_1_ORI_0   = auto()
    ROW_3_COL_1_ORI_45  = auto()
    ROW_3_COL_1_ORI_90  = auto()
    ROW_3_COL_1_ORI_135 = auto()
    ROW_3_COL_1_ORI_180 = auto()
    ROW_3_COL_1_ORI_225 = auto()
    ROW_3_COL_1_ORI_270 = auto()
    ROW_3_COL_1_ORI_315 = auto()

    # Row 3, column 2
    ROW_3_COL_2_ORI_0   = auto()
    ROW_3_COL_2_ORI_45  = auto()
    ROW_3_COL_2_ORI_90  = auto()
    ROW_3_COL_2_ORI_135 = auto()
    ROW_3_COL_2_ORI_180 = auto()
    ROW_3_COL_2_ORI_225 = auto()
    ROW_3_COL_2_ORI_270 = auto()
    ROW_3_COL_2_ORI_315 = auto()

    # Row 3, column 3
    ROW_3_COL_3_ORI_0   = auto()
    ROW_3_COL_3_ORI_45  = auto()
    ROW_3_COL_3_ORI_90  = auto()
    ROW_3_COL_3_ORI_135 = auto()
    ROW_3_COL_3_ORI_180 = auto()
    ROW_3_COL_3_ORI_225 = auto()
    ROW_3_COL_3_ORI_270 = auto()
    ROW_3_COL_3_ORI_315 = auto()

    # Row 3, column 4
    ROW_3_COL_4_ORI_0   = auto()
    ROW_3_COL_4_ORI_45  = auto()
    ROW_3_COL_4_ORI_90  = auto()
    ROW_3_COL_4_ORI_135 = auto()
    ROW_3_COL_4_ORI_180 = auto()
    ROW_3_COL_4_ORI_225 = auto()
    ROW_3_COL_4_ORI_270 = auto()
    ROW_3_COL_4_ORI_315 = auto()

    @property
    def parameters(self):
        """Return (row, column, orientation)."""
        if self == type(self).NONE:
            return None, None, None

        _, row, _, col, _, ori = self.name.split("_")
        return int(row), int(col), int(ori)

    @property
    def row(self):
        return self.parameters[0]

    @property
    def column(self):
        return self.parameters[1]

    @property
    def orientation(self):
        return self.parameters[2]

    @property
    def position(self):
        return self.parameters[:2]

    @property
    def spatial_frequency(self):
        """Temporal frequency in cycles/degree."""
        return None if self is type(self).NONE else 0.04

    @property
    def temporal_frequency(self):
        """Temporal frequency in Hz."""
        return None if self is type(self).NONE else 1.0


@unique
class DriftingGrating(IntEnum):
    """
    Each trial should have exactly one trial type, or NONE.

    We have 12 orientations with 30 deg interval, and 2 spatial frequencies.
    """
    NONE = 0

    # Spatial frequency: 0.04 cycles/degree
    ORI_0_SF_004   = auto() # 1
    ORI_30_SF_004  = auto() # 2
    ORI_60_SF_004  = auto() # 3  
    ORI_90_SF_004  = auto() # 4  
    ORI_120_SF_004 = auto() # 5  
    ORI_150_SF_004 = auto() # 6  
    ORI_180_SF_004 = auto() # 7  
    ORI_210_SF_004 = auto() # 8  
    ORI_240_SF_004 = auto() # 9  
    ORI_270_SF_004 = auto() # 10 
    ORI_300_SF_004 = auto() # 11 
    ORI_330_SF_004 = auto() # 12 

    # Spatial frequency: 0.16 cycles/degree
    ORI_0_SF_016   = auto() # 13
    ORI_30_SF_016  = auto() # 14
    ORI_60_SF_016  = auto() # 15
    ORI_90_SF_016  = auto() # 16
    ORI_120_SF_016 = auto() # 17
    ORI_150_SF_016 = auto() # 18
    ORI_180_SF_016 = auto() # 19
    ORI_210_SF_016 = auto() # 20
    ORI_240_SF_016 = auto() # 21
    ORI_270_SF_016 = auto() # 22
    ORI_300_SF_016 = auto() # 23
    ORI_330_SF_016 = auto() # 24

    @property
    def parameters(self):
        """Return (orientation_degrees, spatial_frequency)."""
        if self == type(self).NONE:
            return None, None

        # Example: "ORI_30_SF_004" -> ["ORI", "30", "SF", "004"]
        _, ori, _, sf = self.name.split("_")

        # Naming convention: 004 means 0.04; 016 means 0.16.
        return int(ori), int(sf) / 100

    @property
    def orientation(self):
        return self.parameters[0]

    @property
    def spatial_frequency(self):
        return self.parameters[1]

    @property
    def temporal_frequency(self):
        """Temporal frequency in Hz."""
        return None if self is type(self).NONE else 1.0


@unique
class Movie(IntEnum):
    """
    Each trial should have exactly one trial type, or NONE.

    We have 2 movies, both 60s.
    """
    NONE = 0

    # cage movie
    cage = auto() # 1

    # natural movie
    natural = auto() # 2

    @property
    def duration(self):
        return None if self is type(self).NONE else 60


@unique
class Events(IntFlag):
    """
    Defines events that could happen during vr sessions.

    Events can be added on top of each other.
    """
    NONE = 0

    trial_start = auto() # 1
    gray_on = auto() # 2
    gray_off = auto() # 4
    dark_on = auto() # 8
    dark_off = auto() # 16
    static_on = auto() # 32
    static_off = auto() # 64
    forward_on = auto() # 128
    forward_off = auto() # 256
    trial_end = auto() # 512


# TODO nov 11 2026 continue here
# each trial we need four values?
