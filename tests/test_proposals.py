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
