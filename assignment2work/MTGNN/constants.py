"""
Module: constants.py
(Added by AI Agent)

This module defines common constants used across the MTGNN models, primarily related to timestamping predictions.
"""

from datetime import datetime

from util import format_time_as_YYYYMMddHHmm

class Constants:
    """
    Class containing constant definitions.
    (Added by AI Agent)

    Attributes:
        PREDICTION_TIME (str): A string representing the current time, formatted as YYYYMMddHHmm.
                               Used to timestamp output files directly via a constant.
    """
    PREDICTION_TIME = format_time_as_YYYYMMddHHmm(datetime.now())
