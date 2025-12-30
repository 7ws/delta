"""Integration tests for acp_server module.

These tests verify that the ACP server imports correctly and can be
instantiated. They catch import errors that unit tests miss when they
test modules in isolation.
"""


class TestAcpServerImports:
    """Verify all imports in acp_server resolve correctly."""

    def test_delta_agent_imports(self) -> None:
        # When
        from delta.acp_server import DeltaAgent

        # Then
        assert DeltaAgent is not None

    def test_run_server_imports(self) -> None:
        # When
        from delta.acp_server import run_server

        # Then
        assert run_server is not None

    def test_compliance_state_imports(self) -> None:
        # When
        from delta.acp_server import ComplianceState

        # Then
        assert ComplianceState is not None

    def test_workflow_phase_imports(self) -> None:
        # When
        from delta.acp_server import WorkflowPhase

        # Then
        assert WorkflowPhase is not None


class TestDeltaAgentInstantiation:
    """Verify DeltaAgent can be instantiated."""

    def test_given_no_args_when_instantiated_then_creates_agent(self) -> None:
        # Given
        from delta.acp_server import DeltaAgent

        # When
        agent = DeltaAgent()

        # Then
        assert agent is not None
        assert agent.sessions == {}

    def test_given_custom_classify_model_when_instantiated_then_uses_model(
        self,
    ) -> None:
        # Given
        from delta.acp_server import DeltaAgent

        # When
        agent = DeltaAgent(classify_model="sonnet")

        # Then
        assert agent.classify_model == "sonnet"


class TestComplianceStateLifecycle:
    """Verify ComplianceState works correctly."""

    def test_given_new_state_when_created_then_has_defaults(self) -> None:
        # Given
        from delta.acp_server import ComplianceState, WorkflowPhase

        # When
        state = ComplianceState()

        # Then
        assert state.phase == WorkflowPhase.PLANNING
        assert state.approved_plan == ""
        assert state.plan_tasks == []
        assert state.tool_call_history == []

    def test_given_state_with_history_when_reset_then_preserves_incomplete_tasks(
        self,
    ) -> None:
        # Given
        from delta.acp_server import ComplianceState, WorkflowPhase
        from delta.plan_widget import PlanTask

        state = ComplianceState()
        state.phase = WorkflowPhase.EXECUTING
        state.approved_plan = "Do something"
        state.plan_tasks = [
            PlanTask(content="Task 1", status="completed"),
            PlanTask(content="Task 2", status="in_progress"),
            PlanTask(content="Task 3", status="pending"),
        ]

        # When
        state.reset_for_new_prompt()

        # Then
        assert state.phase == WorkflowPhase.PLANNING
        assert state.approved_plan == ""
        # Completed tasks removed, incomplete tasks preserved
        assert len(state.plan_tasks) == 2
        assert state.plan_tasks[0].content == "Task 2"
        assert state.plan_tasks[1].content == "Task 3"

    def test_given_state_when_tool_recorded_then_appears_in_history(self) -> None:
        # Given
        from delta.acp_server import ComplianceState

        state = ComplianceState()

        # When
        state.record_tool_call("Write file: /tmp/test.py", allowed=True)
        state.record_tool_call("Delete file: /tmp/secret.py", allowed=False)

        # Then
        assert len(state.tool_call_history) == 2
        assert "[ALLOWED]" in state.tool_call_history[0]
        assert "[DENIED]" in state.tool_call_history[1]
