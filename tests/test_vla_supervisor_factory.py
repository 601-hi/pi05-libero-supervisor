from vla_supervisor.factory import create_safe_default


def test_default_factory_declares_placeholders():
    runtime=create_safe_default()
    assert runtime.component_status["instruction_guard"]=="active"
    assert runtime.component_status["policy_stall"]=="active"
    assert runtime.component_status["execution_consistency"]=="placeholder"
    assert runtime.component_status["object_result"]=="placeholder"
