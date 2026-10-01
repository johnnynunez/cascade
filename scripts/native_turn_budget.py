"""Wall budgets shared by native attendee orders and their acceptance proof.

The turn includes model inference, tool calls and the final response. It is
not a promise of 300 seconds for each tool. The separate MCP call limit and
all robot/search/motion deadlines remain unchanged.
"""

NATIVE_TURN_TIMEOUT_S = 300
AGENT_EXIT_GRACE_S = 30
