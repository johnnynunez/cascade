"""Wall budgets shared by native attendee orders and their acceptance proof.

The turn includes model inference, tool calls and the final response. Reserve
60 seconds for the host around the existing 300-second MCP call limit. This
does not grant extra time to a tool or alter robot/search/motion deadlines.
Multiple tools still share the one bounded turn; expiry still cancels motion.
"""

NATIVE_TURN_TIMEOUT_S = 360
AGENT_EXIT_GRACE_S = 30
