"""Smoke tests for SkyRL-Gym tool group schemas.

Validates:
- All 14 tools have required fields (name, description, parameters)
- Tool schemas are valid OpenAI function format
- get_tool_schemas() returns list of dicts
- get_tool_names() returns list of strings
"""

import pytest

from open_ctf.envs.skyrl.tool_groups import OPENCTF_TOOLS, get_tool_schemas, get_tool_names


# ---------------------------------------------------------------------------
# Tool count and basic structure
# ---------------------------------------------------------------------------


class TestToolGroupBasics:
    def test_fourteen_tools_defined(self):
        assert len(OPENCTF_TOOLS) == 14

    def test_all_are_dicts(self):
        for tool in OPENCTF_TOOLS:
            assert isinstance(tool, dict)

    def test_all_have_type_function(self):
        for tool in OPENCTF_TOOLS:
            assert tool["type"] == "function", f"Tool missing type=function: {tool}"

    def test_all_have_function_key(self):
        for tool in OPENCTF_TOOLS:
            assert "function" in tool


# ---------------------------------------------------------------------------
# OpenAI function schema compliance
# ---------------------------------------------------------------------------


class TestToolSchemaCompliance:
    def test_all_have_name(self):
        for tool in OPENCTF_TOOLS:
            fn = tool["function"]
            assert "name" in fn, f"Tool missing name: {fn}"
            assert isinstance(fn["name"], str)
            assert len(fn["name"]) > 0

    def test_all_have_description(self):
        for tool in OPENCTF_TOOLS:
            fn = tool["function"]
            assert "description" in fn, f"Tool {fn.get('name', '?')} missing description"
            assert isinstance(fn["description"], str)
            assert len(fn["description"]) > 0

    def test_all_have_parameters(self):
        for tool in OPENCTF_TOOLS:
            fn = tool["function"]
            assert "parameters" in fn, f"Tool {fn['name']} missing parameters"

    def test_parameters_type_is_object(self):
        for tool in OPENCTF_TOOLS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object", (
                f"Tool {tool['function']['name']} parameters.type != object"
            )

    def test_parameters_have_properties(self):
        for tool in OPENCTF_TOOLS:
            params = tool["function"]["parameters"]
            assert "properties" in params, (
                f"Tool {tool['function']['name']} missing parameters.properties"
            )

    def test_parameters_have_required(self):
        for tool in OPENCTF_TOOLS:
            params = tool["function"]["parameters"]
            assert "required" in params, (
                f"Tool {tool['function']['name']} missing parameters.required"
            )
            assert isinstance(params["required"], list)

    def test_required_fields_exist_in_properties(self):
        """Every required field should be listed in properties."""
        for tool in OPENCTF_TOOLS:
            fn = tool["function"]
            params = fn["parameters"]
            props = set(params["properties"].keys())
            for req in params["required"]:
                assert req in props, (
                    f"Tool {fn['name']}: required field '{req}' not in properties"
                )


# ---------------------------------------------------------------------------
# Expected tool names
# ---------------------------------------------------------------------------


class TestExpectedTools:
    EXPECTED_NAMES = {
        "shell_command",
        "python_code",
        "read_file",
        "grep",
        "file_search",
        "apply_patch",
        "flag_found",
        "web_search",
        "exec_command",
        "write_stdin",
        "execute_command",
        "list_sessions",
        "close_session",
        "submit_flag",
    }

    def test_all_expected_tools_present(self):
        actual = {t["function"]["name"] for t in OPENCTF_TOOLS}
        for name in self.EXPECTED_NAMES:
            assert name in actual, f"Expected tool '{name}' not found"

    def test_no_unexpected_tools(self):
        actual = {t["function"]["name"] for t in OPENCTF_TOOLS}
        unexpected = actual - self.EXPECTED_NAMES
        assert not unexpected, f"Unexpected tools: {unexpected}"

    def test_shell_command_requires_command(self):
        tool = next(t for t in OPENCTF_TOOLS if t["function"]["name"] == "shell_command")
        assert "command" in tool["function"]["parameters"]["required"]

    def test_flag_found_requires_content(self):
        tool = next(t for t in OPENCTF_TOOLS if t["function"]["name"] == "flag_found")
        assert "content" in tool["function"]["parameters"]["required"]

    def test_python_code_requires_code(self):
        tool = next(t for t in OPENCTF_TOOLS if t["function"]["name"] == "python_code")
        assert "code" in tool["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_get_tool_schemas_returns_list(self):
        result = get_tool_schemas()
        assert isinstance(result, list)
        assert len(result) == 14

    def test_get_tool_schemas_is_openctf_tools(self):
        assert get_tool_schemas() is OPENCTF_TOOLS

    def test_get_tool_names_returns_list(self):
        result = get_tool_names()
        assert isinstance(result, list)
        assert len(result) == 14

    def test_get_tool_names_all_strings(self):
        for name in get_tool_names():
            assert isinstance(name, str)

    def test_get_tool_names_matches_schemas(self):
        names = get_tool_names()
        schema_names = [t["function"]["name"] for t in OPENCTF_TOOLS]
        assert names == schema_names
