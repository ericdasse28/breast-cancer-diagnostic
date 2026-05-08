"""Utilities to assess the performance of the breast cancer diagnostic model."""

from dataclasses import dataclass


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
