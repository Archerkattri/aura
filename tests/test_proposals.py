from types import SimpleNamespace

import pytest

from aura import (
    CaptureFrameTensors,
    CaptureProposalModel,
    CaptureProposalFeatures,
    CaptureProposalTrainingExample,
    CaptureTensor,
    TrainingFrame,
    capture_proposal_features,
    propose_training_regions_from_tensors,
    score_capture_proposals,
    train_capture_proposal_model,
)


def test_capture_proposal_model_scores_tensor_features():
    frame = _frame()
    tensors = _tensors()
    asset = _asset()

    features = capture_proposal_features(frame, tensors, asset)
    scores = score_capture_proposals((frame,), {"frame": tensors}, {"frame": asset})

    assert features.image_detail == pytest.approx(2.0 / 3.0)
    assert features.depth_edge == pytest.approx(0.5)
    assert features.mask_coverage == 0.5
    assert features.normal_present is True
    assert [score.proposal_type for score in scores] == ["image_detail", "depth_edge"]
    assert all(score.accepted for score in scores)
    assert scores[0].to_dict()["modelId"] == "aura-reference-capture-proposal-v1"


def test_capture_proposal_model_threshold_controls_native_regions():
    frame = _frame()
    tensors = _tensors()
    asset = _asset()

    accepted = propose_training_regions_from_tensors(
        (frame,),
        {"frame": tensors},
        {"frame": asset},
        model=CaptureProposalModel(threshold=0.55),
    )
    rejected = propose_training_regions_from_tensors(
        (frame,),
        {"frame": tensors},
        {"frame": asset},
        model=CaptureProposalModel(threshold=0.99),
    )

    assert [region.id for region in accepted] == ["frame_image_detail_proposal", "frame_depth_edge_proposal"]
    assert accepted[0].evidence.high_frequency >= 0.8
    assert accepted[1].evidence.compact_detail >= 0.8
    assert all(region.fallback_source == "capture-feature-proposal" for region in accepted)
    assert rejected == ()


def test_train_capture_proposal_model_learns_labeled_feature_contract():
    examples = (
        CaptureProposalTrainingExample(
            features=CaptureProposalFeatures(
                frame_id="image_positive",
                image_detail=0.95,
                depth_edge=0.05,
                mask_coverage=0.2,
                normal_present=False,
                depth=1.0,
            ),
            image_detail_label=True,
            depth_edge_label=False,
        ),
        CaptureProposalTrainingExample(
            features=CaptureProposalFeatures(
                frame_id="depth_positive",
                image_detail=0.05,
                depth_edge=0.95,
                mask_coverage=0.2,
                normal_present=True,
                depth=1.0,
            ),
            image_detail_label=False,
            depth_edge_label=True,
        ),
        CaptureProposalTrainingExample(
            features=CaptureProposalFeatures(
                frame_id="negative",
                image_detail=0.02,
                depth_edge=0.02,
                mask_coverage=0.0,
                normal_present=False,
                depth=1.0,
            ),
            image_detail_label=False,
            depth_edge_label=False,
            weight=2.0,
        ),
    )

    model = train_capture_proposal_model(examples, iterations=160, learning_rate=1.0, threshold=0.5)
    round_tripped = CaptureProposalModel.from_dict(model.to_dict())
    image_scores = round_tripped.score(examples[0].features)
    depth_scores = round_tripped.score(examples[1].features)
    negative_scores = round_tripped.score(examples[2].features)

    assert round_tripped.id == "aura-learned-capture-proposal-v1"
    assert image_scores[0].accepted is True
    assert image_scores[1].accepted is False
    assert depth_scores[0].accepted is False
    assert depth_scores[1].accepted is True
    assert all(score.accepted is False for score in negative_scores)
    assert model.to_dict()["format"] == "AURA_CAPTURE_PROPOSAL_MODEL"
    assert model.to_dict()["featureOrder"] == ["bias", "image_detail", "depth_edge", "mask_coverage", "normal_present"]


def test_learned_capture_proposal_model_feeds_native_region_generation():
    frame = _frame()
    tensors = _tensors()
    asset = _asset()
    features = capture_proposal_features(frame, tensors, asset)
    model = train_capture_proposal_model(
        (
            CaptureProposalTrainingExample(features=features, image_detail_label=True, depth_edge_label=True),
            CaptureProposalTrainingExample(
                features=CaptureProposalFeatures(
                    frame_id="negative",
                    image_detail=0.0,
                    depth_edge=0.0,
                    mask_coverage=0.0,
                    normal_present=False,
                    depth=2.0,
                ),
                image_detail_label=False,
                depth_edge_label=False,
            ),
        ),
        iterations=120,
        learning_rate=1.0,
        threshold=0.5,
    )

    regions = propose_training_regions_from_tensors(
        (frame,),
        {"frame": tensors},
        {"frame": asset},
        model=model,
    )

    assert [region.id for region in regions] == ["frame_image_detail_proposal", "frame_depth_edge_proposal"]
    assert all(region.fallback_source == "capture-feature-proposal" for region in regions)


def test_capture_proposal_training_validates_input():
    with pytest.raises(ValueError, match="at least one example"):
        train_capture_proposal_model(())
    with pytest.raises(ValueError, match="weight"):
        CaptureProposalTrainingExample(
            features=CaptureProposalFeatures(
                frame_id="bad",
                image_detail=0.0,
                depth_edge=0.0,
                mask_coverage=0.0,
                normal_present=False,
                depth=1.0,
            ),
            image_detail_label=False,
            depth_edge_label=False,
            weight=0.0,
        )
    with pytest.raises(ValueError, match="missing weights"):
        CaptureProposalModel.from_dict(
            {
                "format": "AURA_CAPTURE_PROPOSAL_MODEL",
                "imageDetailWeights": {"bias": 0.0},
                "depthEdgeWeights": {"bias": 0.0},
            }
        )


def _frame() -> TrainingFrame:
    return TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
    )


def _tensors() -> CaptureFrameTensors:
    return CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            path="frame.ppm",
            format="Netpbm",
            backend="stdlib",
            width=2,
            height=1,
            channels=3,
            values=(1.0, 0.0, 0.0, 0.0, 0.5, 0.5),
        ),
        depth=CaptureTensor(
            path="frame.pgm",
            format="Netpbm",
            backend="stdlib",
            width=2,
            height=1,
            channels=1,
            values=(0.5, 1.0),
        ),
    )


def _asset():
    return SimpleNamespace(average_depth=0.75, mask_coverage=0.5, average_normal=(0.0, 0.0, -1.0))


# --- Additional coverage tests ---

def test_capture_proposal_training_example_to_dict():
    """Cover CaptureProposalTrainingExample.to_dict() (line 71)."""
    features = CaptureProposalFeatures(
        frame_id="f1",
        image_detail=0.5,
        depth_edge=0.3,
        mask_coverage=0.2,
        normal_present=True,
        depth=1.0,
    )
    example = CaptureProposalTrainingExample(
        features=features,
        image_detail_label=True,
        depth_edge_label=False,
        weight=1.5,
    )
    d = example.to_dict()
    assert d["imageDetailLabel"] is True
    assert d["depthEdgeLabel"] is False
    assert d["weight"] == 1.5
    assert "features" in d


def test_capture_proposal_model_threshold_out_of_range():
    """Cover CaptureProposalModel.__post_init__ threshold check (line 95)."""
    with pytest.raises(ValueError, match="threshold"):
        CaptureProposalModel(threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        CaptureProposalModel(threshold=-0.1)


def test_capture_proposal_model_from_dict_wrong_format():
    """Cover CaptureProposalModel.from_dict format guard (line 132)."""
    with pytest.raises(ValueError, match="AURA_CAPTURE_PROPOSAL_MODEL"):
        CaptureProposalModel.from_dict({"format": "WRONG_FORMAT"})


def test_train_capture_proposal_model_rejects_invalid_iterations():
    """Cover train_capture_proposal_model validation branches (lines 165, 167, 169)."""
    features = CaptureProposalFeatures(
        frame_id="f", image_detail=0.5, depth_edge=0.5, mask_coverage=0.0, normal_present=False, depth=1.0
    )
    example = CaptureProposalTrainingExample(features=features, image_detail_label=True, depth_edge_label=False)

    with pytest.raises(ValueError, match="iterations must be positive"):
        train_capture_proposal_model((example,), iterations=0)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        train_capture_proposal_model((example,), learning_rate=0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        train_capture_proposal_model((example,), threshold=1.5)


def test_score_capture_proposals_skips_frames_without_tensors():
    """Cover score_capture_proposals when frame has no tensor entry (line 202)."""
    frame = _frame()
    # Pass empty tensors dict — the frame has no entry, so it should be skipped
    scores = score_capture_proposals((frame,), {}, {})
    assert scores == ()


def test_propose_training_regions_skips_frames_without_tensors():
    """Cover propose_training_regions_from_tensors skip path (line 221)."""
    frame = _frame()
    regions = propose_training_regions_from_tensors((frame,), {}, {})
    assert regions == ()


def test_validate_weight_mapping_rejects_non_mapping():
    """Cover _validate_weight_mapping when value is not a Mapping (line 389)."""
    with pytest.raises(ValueError, match="imageDetailWeights must be a mapping"):
        CaptureProposalModel.from_dict(
            {
                "format": "AURA_CAPTURE_PROPOSAL_MODEL",
                "imageDetailWeights": "not_a_mapping",
                "depthEdgeWeights": {"bias": 0.0},
            }
        )


def test_average_tensor_rgb_raises_for_fewer_than_3_channels():
    """Cover _average_tensor_rgb channel check (line 399)."""
    from aura import CaptureTensor
    from aura.proposals import _average_tensor_rgb

    single_channel = CaptureTensor(
        path="x.pgm",
        format="Netpbm",
        backend="stdlib",
        width=2,
        height=1,
        channels=1,
        values=(0.5, 1.0),
    )
    with pytest.raises(ValueError, match="3-channel"):
        _average_tensor_rgb(single_channel)


def test_image_detail_score_returns_zero_for_single_pixel_or_wrong_channels():
    """Cover _image_detail_score early-return branches (lines 410, 417)."""
    from aura import CaptureTensor
    from aura.proposals import _image_detail_score

    # Single channel — returns 0.0 (line 410)
    mono = CaptureTensor(
        path="x.pgm", format="Netpbm", backend="stdlib", width=2, height=1, channels=1, values=(0.5, 0.5)
    )
    assert _image_detail_score(mono) == 0.0

    # 3-channel but only 1 pixel — returns 0.0 (line 410 second condition)
    one_pixel = CaptureTensor(
        path="x.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(0.5, 0.5, 0.5)
    )
    assert _image_detail_score(one_pixel) == 0.0

    # Uniform 3-channel 2x2 image — covers vertical neighbor (line 417 scores.append path)
    uniform_2x2 = CaptureTensor(
        path="u.ppm", format="Netpbm", backend="stdlib", width=2, height=2, channels=3,
        values=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    )
    assert _image_detail_score(uniform_2x2) == 0.0


def test_depth_edge_score_branches():
    """Cover _depth_edge_score branches (lines 429, 435, 441-443)."""
    from aura import CaptureTensor
    from aura.proposals import _depth_edge_score

    # None depth returns 0.0 (line 429)
    assert _depth_edge_score(None) == 0.0

    # wrong channels returns 0.0
    two_chan = CaptureTensor(
        path="x", format="Netpbm", backend="stdlib", width=2, height=1, channels=2, values=(0.5, 0.5, 0.5, 0.5)
    )
    assert _depth_edge_score(two_chan) == 0.0

    # single pixel returns 0.0
    one_pixel_depth = CaptureTensor(
        path="x", format="Netpbm", backend="stdlib", width=1, height=1, channels=1, values=(0.5,)
    )
    assert _depth_edge_score(one_pixel_depth) == 0.0

    # Zero-value pixel is skipped (line 435 continue branch): all zeros => no scores => 0.0
    all_zero = CaptureTensor(
        path="z", format="Netpbm", backend="stdlib", width=2, height=2, channels=1, values=(0.0, 0.0, 0.0, 0.0)
    )
    assert _depth_edge_score(all_zero) == 0.0

    # 2x2 depth map with positive values — exercises horizontal and vertical neighbor checks
    depth_2x2 = CaptureTensor(
        path="d", format="Netpbm", backend="stdlib", width=2, height=2, channels=1, values=(1.0, 2.0, 1.5, 2.5)
    )
    score = _depth_edge_score(depth_2x2)
    assert score >= 0.0

    # Neighbor is zero — line 441-443 neighbor<=0 branch: skip that neighbor
    mixed_zero = CaptureTensor(
        path="m", format="Netpbm", backend="stdlib", width=2, height=1, channels=1, values=(1.0, 0.0)
    )
    # x=0 has valid value but x+1=0 (skipped), no vertical neighbor => scores empty => 0.0
    assert _depth_edge_score(mixed_zero) == 0.0


def test_depth_region_half_extent_with_no_intrinsics():
    """Cover _depth_region_half_extent without intrinsics (line 449)."""
    from aura.proposals import _depth_region_half_extent

    frame_no_intrinsics = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics=None,
    )
    result = _depth_region_half_extent(frame_no_intrinsics, 1.0)
    assert result == pytest.approx(0.05)  # max(0.05, 1.0 * 0.05)


def test_example_label_raises_for_unknown_label():
    """Cover _example_label unknown label branch (line 374)."""
    from aura.proposals import _example_label
    from aura import CaptureProposalFeatures
    features = CaptureProposalFeatures(
        frame_id="f", image_detail=0.5, depth_edge=0.5, mask_coverage=0.0, normal_present=False, depth=1.0
    )
    example = CaptureProposalTrainingExample(features=features, image_detail_label=True, depth_edge_label=False)
    with pytest.raises(ValueError, match="unknown proposal label"):
        _example_label(example, "not_a_real_label")
