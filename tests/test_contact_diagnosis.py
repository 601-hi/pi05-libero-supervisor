from vla_supervisor.contact_diagnosis import (
    ContactDiagnosisConfig,
    ContactEvidence,
    ContactGraspSemanticStateMachine,
    ContactGraspState,
)


def sample(**updates):
    values = dict(
        execution_decision="unknown",
        execution_reliability=0.2,
        manipulation_phase=True,
        gripper_closed=True,
        background_confidence=0.9,
        gripper_candidate_proximity=0.9,
    )
    values.update(updates)
    return ContactEvidence(**values)


def enter_contact(machine):
    machine.update(sample())
    result = machine.update(sample())
    assert result.state is ContactGraspState.CONTACT_SUSPECTED


def test_low_execution_reliability_alone_does_not_imply_contact():
    machine = ContactGraspSemanticStateMachine()
    for _ in range(4):
        result = machine.update(sample(gripper_candidate_proximity=None))
    assert result.state is ContactGraspState.FREE_MOTION


def test_high_candidate_quality_does_not_imply_contact_or_attachment():
    machine = ContactGraspSemanticStateMachine()
    for _ in range(4):
        result = machine.update(sample(
            controlled_object_candidate_probability=.99,
            gripper_candidate_proximity=None,
            independent_motion_probability=None,
            attachment_probability=None,
        ))
    assert result.state is ContactGraspState.FREE_MOTION
    assert result.evidence["controlled_object_candidate_probability"] == .99


def test_clear_execution_abnormality_is_not_relabelled_as_contact():
    machine = ContactGraspSemanticStateMachine()
    for _ in range(4):
        result = machine.update(sample(execution_decision="known_abnormal",
                                       execution_reliability=.95))
    assert result.state is ContactGraspState.FREE_MOTION


def test_future_motion_and_attachment_establish_then_semantically_verify_target():
    machine = ContactGraspSemanticStateMachine()
    enter_contact(machine)
    for _ in range(3):
        result = machine.update(sample(independent_motion_probability=.9,
                                       attachment_probability=.95))
    assert result.state is ContactGraspState.OBJECT_CONTROL_ESTABLISHED
    result = machine.update(sample(semantic_target_probability=.91,
                                   attachment_probability=.95))
    assert result.state is ContactGraspState.TARGET_CONTROLLED


def test_wrong_object_is_actionable_only_after_control_is_established():
    machine = ContactGraspSemanticStateMachine()
    enter_contact(machine)
    for _ in range(3):
        result = machine.update(sample(independent_motion_probability=.9,
                                       attachment_probability=.9,
                                       semantic_target_probability=.05))
    assert result.state is ContactGraspState.OBJECT_CONTROL_ESTABLISHED
    assert not result.actionable
    result = machine.update(sample(semantic_target_probability=.05,
                                   attachment_probability=.9))
    assert result.state is ContactGraspState.WRONG_OBJECT_CONTROL
    assert result.actionable


def test_missing_semantic_score_abstains_after_physical_control():
    machine = ContactGraspSemanticStateMachine()
    enter_contact(machine)
    for _ in range(3):
        result = machine.update(sample(independent_motion_probability=.9,
                                       attachment_probability=.9))
    assert result.state is ContactGraspState.OBJECT_CONTROL_ESTABLISHED
    result = machine.update(sample(semantic_target_probability=None,
                                   attachment_probability=.9))
    assert result.state is ContactGraspState.OBJECT_CONTROL_ESTABLISHED
    assert not result.actionable
    assert result.evidence["reason"] == "semantic_identity_unavailable"


def test_blocked_without_independent_motion_becomes_fixed_obstacle_or_jam():
    machine = ContactGraspSemanticStateMachine()
    enter_contact(machine)
    for _ in range(3):
        result = machine.update(sample(independent_motion_probability=.1,
                                       blocked_motion_probability=.9))
    assert result.state is ContactGraspState.FIXED_OBSTACLE_OR_JAM
    assert result.actionable


def test_no_object_consequence_becomes_empty_grasp_after_observation_window():
    config = ContactDiagnosisConfig(consequence_observation_steps=4)
    machine = ContactGraspSemanticStateMachine(config)
    enter_contact(machine)
    for _ in range(4):
        result = machine.update(sample(independent_motion_probability=.1,
                                       blocked_motion_probability=.1))
    assert result.state is ContactGraspState.EMPTY_GRASP_OR_MISS
    assert result.actionable


def test_attachment_loss_requires_temporal_confirmation():
    machine = ContactGraspSemanticStateMachine()
    enter_contact(machine)
    for _ in range(3):
        machine.update(sample(independent_motion_probability=.9, attachment_probability=.9))
    machine.update(sample(semantic_target_probability=.9, attachment_probability=.9))
    first = machine.update(sample(attachment_probability=.1))
    second = machine.update(sample(attachment_probability=.1))
    third = machine.update(sample(attachment_probability=.1))
    assert first.state is ContactGraspState.TRANSIENT_UNCERTAINTY
    assert second.state is ContactGraspState.TRANSIENT_UNCERTAINTY
    assert third.state is ContactGraspState.OBJECT_LOSS_RISK
    assert third.actionable
