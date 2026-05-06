import shutil
import sys
import unittest
import uuid
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import visualize_world_space_filter as viz


@contextmanager
def _workspace_tmp():
    root = Path.cwd() / f"codex_world_viz_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


LOG_LINES = [
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|5|nose|0.5,1.5,2.5|0.6,1.6,2.6|0.1,0.2,0.3|0.3",
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|2|nose|0.0,1.0,2.0|0.1,1.1,2.1|0.0,0.1,0.2|0.3",
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|2|leftEyeInner|1.0,2.0,3.0|1.1,2.1,3.1|0.0,0.1,0.2|0.3",
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|2|leftEye|2.0,3.0,4.0|2.1,3.1,4.1|0.0,0.1,0.2|0.3",
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|2|leftShoulder|3.0,4.0,5.0|3.1,4.1,5.1|0.0,0.1,0.2|0.3",
    "04-09 16:45:45.779 D/FilterLandMarks(25254): "
    "WORLD_EMA|2|rightShoulder|4.0,5.0,6.0|4.1,5.1,6.1|0.0,0.1,0.2|0.3",
]


def _write_log(root: Path, lines=None) -> Path:
    path = root / "world.log"
    path.write_text("\n".join(lines or LOG_LINES), encoding="utf-8")
    return path


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(tag="WORLD_EMA", frameNo=1, landmark="nose", variant="WORLD_EMA a=0.3",
             alpha=0.3, rawX=0.0, rawY=1.0, rawZ=2.0, filtX=0.1, filtY=1.1, filtZ=2.1),
        dict(tag="WORLD_EMA", frameNo=1, landmark="leftEye", variant="WORLD_EMA a=0.3",
             alpha=0.3, rawX=10.0, rawY=20.0, rawZ=30.0, filtX=10.1, filtY=20.1, filtZ=30.1),
        dict(tag="WORLD_EMA", frameNo=3, landmark="nose", variant="WORLD_EMA a=0.3",
             alpha=0.3, rawX=3.0, rawY=4.0, rawZ=5.0, filtX=3.1, filtY=4.1, filtZ=5.1),
        dict(tag="WORLD_EMA", frameNo=3, landmark="leftEye", variant="WORLD_EMA a=0.3",
             alpha=0.3, rawX=13.0, rawY=23.0, rawZ=33.0, filtX=13.1, filtY=23.1, filtZ=33.1),
    ])


class CountingFilter:
    def __init__(self):
        self.calls = []

    def filter_axis(self, **kwargs):
        self.calls.append(kwargs)
        return viz.FilterResult(
            position=kwargs["measuredPos"] + 1.0,
            velocity=kwargs["previousVel"] + 2.0,
            acceleration=kwargs["previousAccel"] + 3.0,
            covPosPos=kwargs["previousCovPosPos"] + 0.1,
            covPosVel=kwargs["previousCovPosVel"] + 0.2,
            covPosAccel=kwargs["previousCovPosAccel"] + 0.3,
            covVelPos=kwargs["previousCovVelPos"] + 0.4,
            covVelVel=kwargs["previousCovVelVel"] + 0.5,
            covVelAccel=kwargs["previousCovVelAccel"] + 0.6,
            covAccelAccel=kwargs["previousCovAccelAccel"] + 0.7,
            innovationNormSquared=kwargs["lastInnovationNormSquared"] + 0.8,
        )


class WorldSpaceFilterVisualizerTests(unittest.TestCase):
    def test_parse_log_captures_existing_format_and_extra_velocity(self):
        with _workspace_tmp() as root:
            df = viz.parse_log(_write_log(root), extra_names=["vel"])

        nose = df[(df["frameNo"] == 2) & (df["landmark"] == "nose")].iloc[0]
        self.assertEqual(nose["rawX"], 0.0)
        self.assertEqual(nose["filtZ"], 2.1)
        self.assertEqual(nose["alpha"], 0.3)
        self.assertEqual(nose["velX"], 0.0)
        self.assertEqual(nose["velZ"], 0.2)

    def test_frame_grouping_returns_sorted_unique_frames(self):
        with _workspace_tmp() as root:
            df = viz.parse_log(_write_log(root), extra_names=["vel"])

        frames = viz.group_frames(df)
        self.assertEqual([frame["frameNo"] for frame in frames], [2, 5])

    def test_skeleton_segments_create_connections_and_skip_missing_endpoints(self):
        with _workspace_tmp() as root:
            df = viz.parse_log(_write_log(root), extra_names=["vel"])

        frame = viz.group_frames(df)[0]
        segments = viz.build_skeleton_segments(frame["landmarks"], prefix="raw")
        pairs = {(segment["start"], segment["end"]) for segment in segments}

        self.assertIn(("nose", "leftEyeInner"), pairs)
        self.assertIn(("leftEyeInner", "leftEye"), pairs)
        self.assertIn(("leftShoulder", "rightShoulder"), pairs)
        self.assertNotIn(("leftEye", "leftEyeOuter"), pairs)

    def test_logged_playback_builds_raw_filtered_and_both_trace_structures(self):
        with _workspace_tmp() as root:
            df = viz.parse_log(_write_log(root), extra_names=["vel"])

        raw = viz.build_logged_playback_data(df, visibility="raw")[0]["traces"]
        filtered = viz.build_logged_playback_data(df, visibility="filtered")[0]["traces"]
        both = viz.build_logged_playback_data(df, visibility="both")[0]["traces"]

        self.assertEqual([trace["kind"] for trace in raw], ["raw"])
        self.assertEqual([trace["kind"] for trace in filtered], ["filtered"])
        self.assertEqual([trace["kind"] for trace in both], ["raw", "filtered"])

    def test_simulation_calls_filter_after_initialization_and_keeps_axis_state(self):
        df = _sample_df()
        counter = CountingFilter()

        sim = viz.simulate_world_space_filter(
            df,
            settings=viz.PoseDetectionSettings(),
            fps=10.0,
            filter_obj=counter,
        )

        self.assertEqual(len(counter.calls), 6)
        self.assertEqual(
            sorted(call["previousPos"] for call in counter.calls),
            [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        )
        second_nose = sim[(sim["frameNo"] == 3) & (sim["landmark"] == "nose")].iloc[0]
        self.assertEqual(second_nose["simFiltX"], 4.0)
        self.assertEqual(second_nose["simFiltY"], 5.0)
        self.assertEqual(second_nose["simFiltZ"], 6.0)

    def test_parallel_simulation_matches_sequential(self):
        settings = viz.PoseDetectionSettings(Q=0.01, R=0.05, baseProcessNoiseAccel=0.01)
        sequential = viz.simulate_world_space_filter(
            _sample_df(),
            settings=settings,
            fps=10.0,
            parallel=False,
        )
        parallel = viz.simulate_world_space_filter(
            _sample_df(),
            settings=settings,
            fps=10.0,
            parallel=True,
            workers=2,
            backend="thread",
        )

        pd.testing.assert_frame_equal(
            sequential.sort_index(axis=1),
            parallel.sort_index(axis=1),
            check_dtype=False,
        )

    def test_simulation_cache_persists_outputs_in_sqlite(self):
        with _workspace_tmp() as root:
            cache_db = root / "world_filter_cache.sqlite"
            settings = viz.PoseDetectionSettings(Q=0.01, R=0.05, baseProcessNoiseAccel=0.01)
            first = viz.simulate_world_space_filter_cached(
                _sample_df(),
                settings=settings,
                fps=10.0,
                cache_db=cache_db,
                parallel=False,
            )
            second = viz.simulate_world_space_filter_cached(
                _sample_df(),
                settings=settings,
                fps=10.0,
                cache_db=cache_db,
                parallel=True,
                workers=2,
                backend="thread",
            )

            self.assertTrue(cache_db.exists())
            pd.testing.assert_frame_equal(
                first.sort_index(axis=1),
                second.sort_index(axis=1),
                check_dtype=False,
            )
            with sqlite3.connect(cache_db) as conn:
                datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
                outputs = conn.execute(
                    "SELECT COUNT(*) FROM kalman_filter_outputs"
                ).fetchone()[0]
            self.assertEqual(datasets, 1)
            self.assertEqual(outputs, 1)

            changed = viz.simulate_world_space_filter_cached(
                _sample_df(),
                settings=viz.PoseDetectionSettings(
                    Q=0.02, R=0.05, baseProcessNoiseAccel=0.01
                ),
                fps=10.0,
                cache_db=cache_db,
                parallel=False,
            )

            self.assertEqual(len(changed), len(first))
            with sqlite3.connect(cache_db) as conn:
                outputs = conn.execute(
                    "SELECT COUNT(*) FROM kalman_filter_outputs"
                ).fetchone()[0]
            self.assertEqual(outputs, 2)

    def test_dt_calculation_uses_frame_gaps_and_fps(self):
        self.assertAlmostEqual(viz.compute_dt(None, 10, 20.0), 0.05)
        self.assertAlmostEqual(viz.compute_dt(10, 13, 10.0), 0.3)
        self.assertAlmostEqual(viz.compute_dt(10, 10, 10.0), 0.1)

        counter = CountingFilter()
        viz.simulate_world_space_filter(
            _sample_df(),
            settings=viz.PoseDetectionSettings(),
            fps=10.0,
            filter_obj=counter,
        )
        self.assertTrue(all(call["dt"] == 0.2 for call in counter.calls))

    def test_settings_round_trip_from_string_dict(self):
        settings = viz.parse_pose_detection_settings({
            "Q": "-1.25",
            "R": "2e-3",
            "numPreviousLocations": "7",
            "worldSpaceEMA": "0.42",
        })

        self.assertEqual(settings.Q, -1.25)
        self.assertEqual(settings.R, 0.002)
        self.assertEqual(settings.numPreviousLocations, 7)
        self.assertEqual(settings.worldSpaceEMA, 0.42)
        self.assertEqual(viz.parse_pose_detection_settings(viz.asdict(settings)), settings)

    def test_extract_settings_presets_from_kotlin_json_and_csv(self):
        kotlin = """
        object FilterConfigurations {
            @JvmField val SUPER_SPEED = FilterSettings(
                poseDetection = PoseDetectionSettings(
                    Q = 0.85f,
                    R = 0.0020f,
                    numPreviousLocations = 18,
                    velocityAlphaMin = 0.01f,
                    confidenceRScaling = 300f,
                    worldSpaceEMA = 0.50f
                )
            )
            @JvmField val QUALITY = FilterSettings(
                poseDetection = PoseDetectionSettings(
                    Q = 0.2f,
                    R = 0.055f,
                    numPreviousLocations = 75,
                    positionLockThreshold = 4f,
                    worldSpaceEMA = 0.35f
                )
            )
        }
        """
        kotlin_presets = viz.extract_pose_detection_settings_presets(kotlin)
        self.assertEqual(kotlin_presets["SUPER_SPEED"]["Q"], "0.85")
        self.assertEqual(kotlin_presets["SUPER_SPEED"]["numPreviousLocations"], "18")
        self.assertEqual(kotlin_presets["QUALITY"]["positionLockThreshold"], "4")

        json_presets = viz.extract_pose_detection_settings_presets(
            '{"BALANCED": {"poseDetection": {"Q": 0.344, "R": "0.045", '
            '"numPreviousLocations": 60}}}'
        )
        self.assertEqual(json_presets["BALANCED"]["Q"], "0.344")

        csv_presets = viz.extract_pose_detection_settings_presets(
            "preset,name,value\nFAST,Q,0.5\nFAST,R,0.01\nFAST,numPreviousLocations,42\n"
        )
        csv_settings = viz.parse_pose_detection_settings(csv_presets["FAST"])
        self.assertEqual(csv_settings.Q, 0.5)
        self.assertEqual(csv_settings.R, 0.01)
        self.assertEqual(csv_settings.numPreviousLocations, 42)

    def test_world_space_kalman_filter_runs_android_style_update(self):
        settings = viz.PoseDetectionSettings(Q=0.01, R=0.05, baseProcessNoiseAccel=0.01)
        filter_obj = viz.WorldSpaceKalmanFilter(settings)

        result = filter_obj.filter_axis(
            measuredPos=1.2,
            dt=1.0 / 30.0,
            previousPos=1.0,
            previousVel=0.1,
            previousAccel=0.0,
            previousCovPosPos=1.0,
            previousCovPosVel=0.0,
            previousCovPosAccel=0.0,
            previousCovVelPos=0.0,
            previousCovVelVel=1.0,
            previousCovVelAccel=0.0,
            previousCovAccelPos=0.0,
            previousCovAccelVel=0.0,
            previousCovAccelAccel=1.0,
            baseProcessNoisePos=settings.Q,
            baseProcessNoiseVel=settings.Q,
            baseProcessNoiseAccel=settings.baseProcessNoiseAccel,
            baseMeasurementNoise=settings.R,
            adaptiveQScalingFactor=settings.adaptiveQScalingFactor,
            dynamicRInnovationSlope=settings.dynamicRInnovationSlope,
            dynamicRInnovationThreshold=settings.dynamicRInnovationThreshold,
            lastInnovationNormSquared=0.0,
            positionAlphaMin=settings.positionAlphaMin,
            positionAlphaMax=settings.positionAlphaMax,
            positionAlphaDecay=settings.positionAlphaDecay,
            velocityAlphaMin=settings.velocityAlphaMin,
            velocityAlphaMax=settings.velocityAlphaMax,
            velocityAlphaMidpoint=settings.velocityAlphaMidpoint,
            velocityAlphaSlope=settings.velocityAlphaSlope,
            accelerationAlphaMin=settings.accelerationAlphaMin,
            accelerationAlphaMax=settings.accelerationAlphaMax,
            accelerationAlphaMidpoint=settings.accelerationAlphaMidpoint,
            accelerationAlphaSlope=settings.accelerationAlphaSlope,
            accelerationAlphaDecay=settings.accelerationAlphaDecay,
            accelerationDecayRate=settings.accelerationDecayRate,
            smoothingDecayTimeConstant=settings.smoothingDecayTimeConstant,
            updateDecayTimeConstant=settings.updateDecayTimeConstant,
            accelerationBoundsDecayFactor=settings.accelerationBoundsDecayFactor,
            accelerationInnovationContributionFactor=(
                settings.accelerationInnovationContributionFactor
            ),
            accelerationBoundsThreshold=settings.accelerationBoundsThreshold,
            accelerationSaturationThreshold=settings.accelerationSaturationThreshold,
            accelerationSaturationNoiseReduction=(
                settings.accelerationSaturationNoiseReduction
            ),
            accelerationAdaptiveScalingFactor=settings.accelerationAdaptiveScalingFactor,
            gainAttenuationRate=settings.gainAttenuationRate,
            positionLockThreshold=settings.positionLockThreshold,
            bypassCoercion=False,
        )

        for value in viz.asdict(result).values():
            self.assertTrue(math.isfinite(value))
        self.assertNotEqual(result.position, 1.2)
        self.assertGreater(result.covPosPos, 0.0)
        self.assertGreaterEqual(result.innovationNormSquared, 0.0)

    def test_generated_html_contains_skeleton_controls_and_coefficients(self):
        with _workspace_tmp() as root:
            df = viz.enrich(viz.parse_log(_write_log(root), extra_names=["vel"]), 1.5, 0.0)
            summary = viz.summarize(df)
            out = root / "report.html"
            viz.write_html(
                df,
                summary,
                sorted(set(zip(df["landmark"], df["variant"]))),
                sorted(df["landmark"].unique()),
                out,
                iqr_k=1.5,
                min_delta=0.0,
                extras_label="velX, velY, velZ",
                render_group=False,
                fps=30.0,
                settings=viz.PoseDetectionSettings(),
            )

            html = out.read_text(encoding="utf-8")

        self.assertIn("Skeleton view", html)
        self.assertIn("id='skel-play'", html)
        self.assertIn("id='skel-pause'", html)
        self.assertIn("id='skel-show-raw'", html)
        self.assertIn("id='skel-show-filtered'", html)
        self.assertIn("id='skel-show-labels'", html)
        self.assertIn("id='skel-variant-select'", html)
        self.assertIn("id='skel-speed'", html)
        self.assertIn("id='skel-frame-step'", html)
        self.assertIn("id='skel-space'", html)
        self.assertIn("Face lock", html)
        self.assertIn("id='skel-subject-yaw'", html)
        self.assertIn("id='skel-twirl'", html)
        self.assertIn("id='skel-stop-twirl'", html)
        self.assertIn("id='skel-frame-slider'", html)
        self.assertIn("id='skel-frame-count'", html)
        self.assertIn("id='skel-reset-camera'", html)
        self.assertIn("skeleton-variant-block active", html)
        self.assertIn("setSkeletonVariant", html)
        self.assertIn("seekSkeletonFrame", html)
        self.assertIn("updateSkeletonFrameControls", html)
        self.assertIn("setSkeletonSpaceMode", html)
        self.assertIn("setSkeletonSubjectYaw", html)
        self.assertIn("startSkeletonSubjectSpin", html)
        self.assertIn("stopSkeletonSubjectSpin", html)
        self.assertIn("transformedSkeletonFrameData", html)
        self.assertIn("skeletonTransformFromData", html)
        self.assertIn("runWithSkeletonViewState", html)
        self.assertIn("rememberSkeletonViewState", html)
        self.assertIn("captureSkeletonViewState", html)
        self.assertIn("scrollZoom: true", html)
        self.assertIn("orthographic", html)
        self.assertIn("rotateSkeletonCamera", html)
        self.assertIn("panSkeletonRanges", html)
        self.assertIn("position: sticky", html)
        self.assertIn("id='skeleton-log'", html)
        self.assertIn("id='skel-frame-status'", html)
        self.assertIn("skeleton-key raw", html)
        self.assertIn("skeleton-key logged", html)
        self.assertIn("skeleton-key sim", html)
        self.assertIn("clearSkeletonLog", html)
        self.assertIn("skeletonFrameNames", html)
        self.assertIn("skeletonFramePayloads", html)
        self.assertIn("restyleSkeletonFrame", html)
        self.assertIn("manual restyle loop", html)
        self.assertIn("id='coef-Q'", html)
        self.assertIn("id='coef-R'", html)
        self.assertIn("id='coef-numPreviousLocations'", html)

    def test_skeleton_figure_uses_fixed_scene_ranges(self):
        fig = viz.build_skeleton_figure(_sample_df(), "WORLD_EMA a=0.3", fps=30.0)

        self.assertEqual(fig.layout.uirevision, "skeleton-fixed-scene")
        self.assertEqual(fig.layout.scene.uirevision, "skeleton-fixed-scene")
        self.assertFalse(fig.layout.scene.xaxis.autorange)
        self.assertFalse(fig.layout.scene.yaxis.autorange)
        self.assertFalse(fig.layout.scene.zaxis.autorange)
        self.assertEqual(fig.layout.scene.aspectmode, "manual")
        self.assertEqual(fig.layout.scene.xaxis.title.text, "X floor (m)")
        self.assertEqual(fig.layout.scene.yaxis.title.text, "Y height (m)")
        self.assertEqual(fig.layout.scene.zaxis.title.text, "Z depth (m)")
        self.assertEqual(fig.layout.scene.camera.up.y, 1)
        self.assertEqual(fig.layout.scene.camera.projection.type, "orthographic")

    def test_skeleton_payload_maps_world_y_to_y_up_display_height(self):
        frame = viz.group_frames(_sample_df())[0]
        payload = viz._skeleton_plot_payload(frame["landmarks"], "raw")
        nose_index = payload["points"]["text"].index("nose")

        self.assertEqual(payload["points"]["x"][nose_index], 0.0)
        self.assertEqual(payload["points"]["y"][nose_index], -1.0)
        self.assertEqual(payload["points"]["z"][nose_index], 2.0)

    def test_skeleton_figure_uses_distinct_raw_logged_and_simulated_colors(self):
        fig = viz.build_skeleton_figure(_sample_df(), "WORLD_EMA a=0.3", fps=30.0)
        colors = [
            fig.data[0].line.color,
            fig.data[2].line.color,
            fig.data[4].line.color,
        ]

        self.assertEqual(colors[0], viz.SKELETON_RAW_COLOR)
        self.assertEqual(colors[1], viz.SKELETON_LOGGED_FILTERED_COLOR)
        self.assertEqual(colors[2], viz.SKELETON_SIM_FILTERED_COLOR)
        self.assertEqual(len(set(colors)), 3)

    def test_default_output_paths_for_log_fill_report_outputs(self):
        html_path, frames_path, summary_path = viz.default_output_paths_for_log(
            Path("C:/logs/world_ema.txt")
        )

        self.assertEqual(html_path.name, "world_ema_world_filter_review.html")
        self.assertEqual(frames_path.name, "world_ema_world_filter_review_frames.csv")
        self.assertEqual(summary_path.name, "world_ema_world_filter_review_summary.csv")

    def test_cli_smoke_writes_html_frames_and_summary_csv(self):
        with _workspace_tmp() as root:
            log_path = _write_log(root)
            html_path = root / "world.html"
            frames_path = root / "frames.csv"
            summary_path = root / "summary.csv"

            argv = [
                "visualize_world_space_filter.py",
                str(log_path),
                "--out", str(html_path),
                "--csv-frames", str(frames_path),
                "--csv-summary", str(summary_path),
                "--schema", "vel",
                "--fps", "12.5",
                "--no-group",
            ]
            with patch.object(sys, "argv", argv):
                viz.main()

            self.assertTrue(html_path.exists())
            self.assertTrue(frames_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertIn("Skeleton view", html_path.read_text(encoding="utf-8"))

    def test_generate_report_emits_progress_messages(self):
        messages = []
        with _workspace_tmp() as root:
            result = viz.generate_report(
                logfile=_write_log(root),
                out=root / "world.html",
                schema="vel",
                render_group=False,
                fps=12.5,
                settings=viz.PoseDetectionSettings(),
                progress_callback=messages.append,
            )
            self.assertTrue(result["html"].exists())
        self.assertTrue(any("Parsing log file" in message for message in messages))
        self.assertTrue(any("Building Skeleton playback view" in message for message in messages))
        self.assertTrue(any("Finished report" in message for message in messages))

    def test_main_without_logfile_launches_gui(self):
        with patch.object(sys, "argv", ["visualize_world_space_filter.py"]):
            with patch.object(viz, "launch_gui") as launch_gui:
                viz.main()

        launch_gui.assert_called_once()


if __name__ == "__main__":
    unittest.main()
