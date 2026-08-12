"""
User-behavior segmentation (clustering).

Segments browsing sessions into distinct behavioral clusters using
KMeans, producing the labeled ``session_segments`` dataset consumed by
the recommendation engine and dashboard.
"""

from src.clustering.pipeline import ClusteringPipeline
from src.clustering.preprocessor import ClusteringPreprocessor
from src.clustering.profiler import SegmentProfiler
from src.clustering.segmenter import (
    ClusterAnalysis,
    KMeansSegmenter,
    SegmentResult,
)

__all__ = [
    "ClusteringPipeline",
    "ClusteringPreprocessor",
    "ClusterAnalysis",
    "KMeansSegmenter",
    "SegmentProfiler",
    "SegmentResult",
]
