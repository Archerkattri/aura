from types import SimpleNamespace

import pytest

from aura import (
    CaptureFrameTensors,
    CaptureProposalModel,
    CaptureTensor,
    TrainingFrame,
    capture_proposal_features,
    propose_training_regions_from_tensors,
    score_capture_proposals,
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
