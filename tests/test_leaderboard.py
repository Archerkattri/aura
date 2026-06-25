from aura.leaderboard import (
    LeaderboardMetric,
    LeaderboardReport,
    LeaderboardRun,
    MethodSpec,
    SceneSpec,
)


def test_leaderboard_report_blocks_missing_scene_from_readiness():
    scenes = (
        SceneSpec(scene_id="truck", dataset="Tanks and Temples", split="llffhold8", image_scale="native"),
        SceneSpec(scene_id="room", dataset="Mip-NeRF 360", split="llffhold8", image_scale="images_2"),
    )
    baseline = MethodSpec(method_id="aura_beta", role="baseline", backend="dbs", command="existing")
    candidate = MethodSpec(method_id="gsplat_main_mcmc", role="candidate", backend="gsplat-main", command="pending")
    report = LeaderboardReport(
        benchmark_id="aura_sota_v1",
        task="novel_view_synthesis",
        scenes=scenes,
        methods=(baseline, candidate),
        runs=(
            LeaderboardRun(
                scene_id="truck",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 26.0, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="gsplat_main_mcmc",
                metrics=(LeaderboardMetric("psnr", 26.5, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
        ),
        primary_metric="psnr",
    )

    payload = report.to_dict()

    assert payload["leaderboardReady"] is False
    assert payload["missingScenes"] == ["room"]
    assert payload["claimBoundary"]["cannotClaim"]


def test_leaderboard_report_promotes_only_measured_candidate_with_artifact():
    scene = SceneSpec(scene_id="truck", dataset="Tanks and Temples", split="llffhold8", image_scale="native")
    report = LeaderboardReport(
        benchmark_id="aura_sota_v1",
        task="novel_view_synthesis",
        scenes=(scene,),
        methods=(
            MethodSpec(method_id="aura_beta", role="baseline", backend="dbs", command="existing"),
            MethodSpec(method_id="candidate_fixture", role="candidate", backend="fixture", command="none"),
            MethodSpec(method_id="candidate_real", role="candidate", backend="gsplat-main", command="python train.py"),
        ),
        runs=(
            LeaderboardRun(
                scene_id="truck",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 26.0, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="candidate_fixture",
                metrics=(LeaderboardMetric("psnr", 99.0, higher_is_better=True),),
                artifacts=(),
                measured=False,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="candidate_real",
                metrics=(LeaderboardMetric("psnr", 26.5, higher_is_better=True),),
                artifacts=("experiments/results/real_candidate.json",),
                measured=True,
            ),
        ),
        primary_metric="psnr",
    )

    payload = report.to_dict()

    assert payload["leaderboardReady"] is True
    assert payload["promotedMethodIds"] == ["candidate_real"]
    assert payload["comparisons"][0]["winnerMethodId"] == "candidate_real"


def test_leaderboard_report_blocks_partial_scene_promotion_from_readiness():
    scenes = (
        SceneSpec(scene_id="truck", dataset="Tanks and Temples", split="llffhold8", image_scale="native"),
        SceneSpec(scene_id="room", dataset="Mip-NeRF 360", split="llffhold8", image_scale="images_2"),
    )
    report = LeaderboardReport(
        benchmark_id="aura_sota_v1",
        task="novel_view_synthesis",
        scenes=scenes,
        methods=(
            MethodSpec(method_id="aura_beta", role="baseline", backend="dbs", command="existing"),
            MethodSpec(method_id="candidate_real", role="candidate", backend="gsplat-main", command="python train.py"),
        ),
        runs=(
            LeaderboardRun(
                scene_id="truck",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 26.0, higher_is_better=True),),
                artifacts=("experiments/results/baseline_truck.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="candidate_real",
                metrics=(LeaderboardMetric("psnr", 26.5, higher_is_better=True),),
                artifacts=("experiments/results/candidate_truck.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="room",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 31.0, higher_is_better=True),),
                artifacts=("experiments/results/baseline_room.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="room",
                method_id="candidate_real",
                metrics=(LeaderboardMetric("psnr", 30.5, higher_is_better=True),),
                artifacts=("experiments/results/candidate_room.json",),
                measured=True,
            ),
        ),
        primary_metric="psnr",
    )

    payload = report.to_dict()

    assert payload["leaderboardReady"] is False
    assert payload["promotedMethodIds"] == ["candidate_real"]
    assert [row["promoted"] for row in payload["comparisons"]] == [True, False]
