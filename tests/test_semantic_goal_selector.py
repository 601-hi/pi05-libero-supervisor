from vla_supervisor.semantic_goal_selector import SemanticCandidateEvidence, SemanticGoalSelector


def evidence(category, relation, confidence=1.0, observable=True):
    return SemanticCandidateEvidence(category, relation, confidence, observable)


def test_right_category_wrong_drawer_is_rejected():
    decision = SemanticGoalSelector().select(
        {"lower_bowl": evidence(0.95, 0.1)}, relation_required=True
    )
    assert decision.target_candidate_id is None
    assert decision.state == "unobservable_or_unsupported"


def test_category_and_source_relation_must_agree():
    decision = SemanticGoalSelector().select(
        {
            "upper_bowl": evidence(0.9, 0.9),
            "lower_bowl": evidence(0.92, 0.2),
        },
        relation_required=True,
    )
    assert decision.target_candidate_id == "upper_bowl"


def test_close_scores_abstain_instead_of_forcing_choice():
    decision = SemanticGoalSelector(minimum_margin=0.1).select(
        {1: evidence(0.8, 0.8), 2: evidence(0.78, 0.8)}, relation_required=True
    )
    assert decision.target_candidate_id is None
    assert decision.state == "ambiguous"


def test_missing_relation_is_rejected_when_language_requires_it():
    decision = SemanticGoalSelector().select(
        {1: evidence(0.95, None)}, relation_required=True
    )
    assert decision.target_candidate_id is None


def test_relation_can_be_omitted_for_category_only_goal():
    decision = SemanticGoalSelector().select(
        {1: evidence(0.9, None), 2: evidence(0.2, None)}, relation_required=False
    )
    assert decision.target_candidate_id == 1

