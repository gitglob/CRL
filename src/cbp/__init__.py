"""Continual backpropagation, optimizer resets, and interaction diagnostics."""

from .algorithm import ContinualBackprop, attach
from .diagnostics import FeatureProbe, InteractionDiagnostics, plasticity
from .optimizer import AdamCBP
