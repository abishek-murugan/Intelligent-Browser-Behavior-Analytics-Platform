"""
Clustering pipeline.

Orchestrates the full user-behavior segmentation workflow:

    1. Load the gold session features dataset.
    2. Preprocess (select + scale features, fit PCA).
    3. Sweep cluster counts and select the optimal k.
    4. Fit the final KMeans model and assign segments.
    5. Persist the labeled dataset and segment profiles.
    6. Render report plots and log everything to MLflow.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.preprocessor import ClusteringPreprocessor
from src.clustering.profiler import SegmentProfiler
from src.clustering.segmenter import KMeansSegmenter, SegmentResult
from src.clustering.visualizer import (
    plot_elbow,
    plot_pca_scatter,
    plot_profile_heatmap,
    plot_silhouette_curve,
)
from src.exceptions import (
    DataValidationError,
    FileReadError,
)
from src.utils.config_loader import get_config, get_models, get_paths
from src.utils.logger import get_logger
from src.utils.mlflow_utils import (
    get_experiment_id,
    get_experiment_name,
    setup_mlflow,
)

logger = get_logger(__name__)


class ClusteringPipeline:
    """Run the complete session segmentation workflow."""

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        report_dir: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        kmeans_path: str | Path | None = None,
        preprocessor: ClusteringPreprocessor | None = None,
        segmenter: KMeansSegmenter | None = None,
        profiler: SegmentProfiler | None = None,
        track_mlflow: bool = True,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        input_path:
            Gold session features dataset path.

        output_path:
            Destination for the labeled segment dataset.

        report_dir:
            Directory for clustering reports and plots.

        preprocessor:
            Feature preprocessor instance.

        segmenter:
            KMeans segmenter instance.

        profiler:
            Segment profiler instance.

        track_mlflow:
            Whether to log the run to MLflow.
        """

        paths = get_paths()["paths"]
        models = get_models()
        config = get_config()

        random_state = config["project"].get(
            "random_seed",
            42,
        )

        self.input_path = Path(
            input_path
            if input_path is not None
            else paths.get(
                "session_features_gold",
                "data/gold/session_features.parquet",
            )
        ).expanduser()

        self.output_path = Path(
            output_path
            if output_path is not None
            else paths.get(
                "session_segments",
                "data/gold/session_segments.parquet",
            )
        ).expanduser()

        self.report_dir = Path(
            report_dir
            if report_dir is not None
            else paths.get(
                "clustering_reports",
                "reports/clustering",
            )
        ).expanduser()

        self.preprocessor_path = Path(
            preprocessor_path
            if preprocessor_path is not None
            else paths.get(
                "clustering_preprocessor",
                "models/clustering_preprocessor.pkl",
            )
        ).expanduser()

        self.kmeans_path = Path(
            kmeans_path
            if kmeans_path is not None
            else paths.get(
                "kmeans_model",
                "models/kmeans_model.pkl",
            )
        ).expanduser()

        self.track_mlflow = track_mlflow

        clustering = models["machine_learning"]["clustering"]

        self.preprocessor = preprocessor or ClusteringPreprocessor(
            exclude_columns=clustering["preprocess"]["exclude_columns"],
            pca_variance_ratio=clustering["preprocess"]["pca_variance_ratio"],
            random_state=random_state,
        )

        self.segmenter = segmenter or KMeansSegmenter(
            min_clusters=clustering["kmeans"]["min_clusters"],
            max_clusters=clustering["kmeans"]["max_clusters"],
            n_init=clustering["kmeans"]["n_init"],
            random_state=clustering["kmeans"].get(
                "random_state",
                random_state,
            ),
        )

        self.profiler = profiler or SegmentProfiler()

        self.analysis = None

        self.result: SegmentResult | None = None

    def run(self) -> dict[str, object]:
        """
        Execute the complete clustering workflow.

        Returns
        -------
        dict[str, object]
            The labeled dataset, segment profile and generated plot
            paths.
        """

        logger.info("Starting clustering pipeline.")

        dataframe = self._load_input()

        matrix = self.preprocessor.fit_transform(dataframe)

        analysis = self.segmenter.sweep(matrix)

        result = self.segmenter.fit(matrix, k=analysis.optimal_k)

        self.analysis = analysis
        self.result = result

        segments = self._build_segment_dataset(
            dataframe,
            result,
        )

        profile = self.profiler.profile(segments)

        plots = self._render_plots(
            analysis,
            matrix,
            result,
            profile,
        )

        self._persist(
            dataframe,
            segments,
            profile,
            matrix,
            result,
        )

        self._log_mlflow(
            dataframe,
            analysis,
            result,
            profile,
            plots,
        )

        logger.info("Clustering pipeline completed.")

        return {
            "segments": segments,
            "profile": profile,
            "plots": plots,
            "optimal_k": analysis.optimal_k,
            "analysis": analysis,
            "result": result,
        }

    def _load_input(self) -> pd.DataFrame:
        """
        Load and validate the gold session features dataset.
        """

        if not self.input_path.is_file():
            raise FileReadError(f"Session features dataset not found: {self.input_path}")

        try:
            dataframe = pd.read_parquet(self.input_path)

        except (OSError, ImportError) as exc:
            raise FileReadError(
                f"Unable to read session features dataset: {self.input_path}"
            ) from exc

        if dataframe.empty:
            raise DataValidationError(f"Session features dataset is empty: {self.input_path}")

        return dataframe

    def _build_segment_dataset(
        self,
        dataframe: pd.DataFrame,
        result: SegmentResult,
    ) -> pd.DataFrame:
        """
        Attach segment labels and centroid distances to the features.
        """

        segments = dataframe.copy()

        segments["segment_id"] = result.labels

        segments["distance_to_centroid"] = result.distances[
            np.arange(len(result.labels)),
            result.labels,
        ]

        logger.info(
            "Segments assigned | sessions=%d | segments=%d",
            len(segments),
            result.k,
        )

        return segments

    def _render_plots(
        self,
        analysis,
        matrix: np.ndarray,
        result: SegmentResult,
        profile: pd.DataFrame,
    ) -> list[Path]:
        """
        Render and persist the clustering report plots.
        """

        plots = [
            plot_elbow(
                analysis,
                self.report_dir / "elbow.png",
            ),
            plot_silhouette_curve(
                analysis,
                self.report_dir / "silhouette.png",
            ),
        ]

        pca_coordinates = None

        if self.preprocessor.pca is not None:
            pca_coordinates = self.preprocessor.project(matrix)

            plots.append(
                plot_pca_scatter(
                    pca_coordinates,
                    result.labels,
                    result.k,
                    self.report_dir / "pca_segments.png",
                )
            )

        plots.append(
            plot_profile_heatmap(
                profile,
                self.preprocessor.feature_columns,
                self.report_dir / "profile_heatmap.png",
            )
        )

        return plots

    def _persist(
        self,
        dataframe: pd.DataFrame,
        segments: pd.DataFrame,
        profile: pd.DataFrame,
        matrix: np.ndarray,
        result: SegmentResult,
    ) -> None:
        """
        Save dataset and preprocessor artifacts to disk.
        """

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        segments.to_parquet(
            self.output_path,
            index=False,
        )

        logger.info(
            "Segment dataset saved: %s | rows=%d",
            self.output_path,
            len(segments),
        )

        self.profiler.save(
            profile,
            self.report_dir / "segment_profile.csv",
        )

        preprocessor_path = self.preprocessor_path
        kmeans_path = self.kmeans_path
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

        self.preprocessor.save(preprocessor_path)

        with kmeans_path.open("wb") as file:
            pickle.dump(result.model, file)

        logger.info("KMeans model saved to %s", preprocessor_path.parent)

    def _log_mlflow(
        self,
        dataframe: pd.DataFrame,
        analysis,
        result: SegmentResult,
        profile: pd.DataFrame,
        plots: list[Path],
    ) -> None:
        """
        Log the clustering experiment to MLflow.
        """

        if not self.track_mlflow:
            logger.info("MLflow tracking disabled; skipping run.")

            return None

        import mlflow
        import mlflow.sklearn

        setup_mlflow()

        experiment_id = get_experiment_id(get_experiment_name("clustering"))

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"kmeans-k{result.k}",
        ) as run:
            mlflow.log_param("k", result.k)
            mlflow.log_param("n_features", len(self.preprocessor.feature_columns))
            mlflow.log_param("n_components", self.preprocessor.n_components)
            mlflow.log_param(
                "feature_columns",
                ",".join(self.preprocessor.feature_columns),
            )
            mlflow.log_param("random_state", self.segmenter.random_state)
            mlflow.log_param("n_sessions", len(dataframe))

            mlflow.log_metric("inertia", result.inertia)
            mlflow.log_metric(
                "silhouette_score",
                analysis.silhouette[analysis.best_index()],
            )
            mlflow.log_metric(
                "davies_bouldin_score",
                analysis.davies_bouldin[analysis.best_index()],
            )
            mlflow.log_metric(
                "calinski_harabasz_score",
                analysis.calinski_harabasz[analysis.best_index()],
            )

            try:
                mlflow.sklearn.log_model(
                    result.model,
                    "kmeans_model",
                    registered_model_name="BrowserSessionKMeansClusterer",
                )
            except Exception as err:
                logger.warning(
                    "Could not register KMeans model to MLflow registry: %s",
                    err,
                )
                mlflow.sklearn.log_model(result.model, "kmeans_model")
            mlflow.sklearn.log_model(
                self.preprocessor.scaler,
                "scaler",
            )
            mlflow.sklearn.log_model(
                self.preprocessor.pca,
                "pca",
            )

            for plot in plots:
                mlflow.log_artifact(str(plot))

            profile.to_csv(
                self.report_dir / "_mlflow_profile.csv",
                index=False,
            )
            mlflow.log_artifact(str(self.report_dir / "_mlflow_profile.csv"))

            (self.report_dir / "_mlflow_profile.csv").unlink(missing_ok=True)

            logger.info("MLflow run logged | run_id=%s", run.info.run_id)

        return run.info.run_id
