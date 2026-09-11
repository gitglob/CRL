"""Continual backpropagation, optimizer resets, and fixed-input diagnostics."""

from .algorithm import ContinualBackprop, attach
from .diagnostics import FeatureProbe, plasticity
from .optimizer import AdamCBP
