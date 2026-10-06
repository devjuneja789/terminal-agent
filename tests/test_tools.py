from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import Toolbox


class ToolboxTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="terminal-slm-test-")
        self.root = Path(self.temporary.name) / "workspace"
        self.root.mkdir()
        self.toolbox = Toolbox(self.root)

    def tearDown(self) -> None:
        self.toolbox.close()
        self.temporary.cleanup()


class WorkspaceTests(unittest.TestCase):
    def test_default_workspace_is_temporary_and_not_home(self) -> None:
        with patch.dict(os.environ, {"TMPDIR": str(Path.home())}):
            toolbox = Toolbox()
        try:
            self.assertNotEqual(toolbox.workspace_root, Path.home().resolve())
            self.assertTrue(toolbox.workspace_root.is_dir())
        finally:
            root = toolbox.workspace_root
            toolbox.close()
            self.assertFalse(root.exists())

    def test_home_directory_cannot_be_used_as_workspace(self) -> None:
        with self.assertRaises(ValueError):
            Toolbox(Path.home())

    def test_internal_tool_directory_cannot_be_a_symlink(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="terminal-slm-link-test-")
        try:
            root = Path(temporary.name) / "workspace"
            root.mkdir()
            outside = Path(temporary.name) / "outside"
            outside.mkdir()
            (root / ".terminal-slm").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                Toolbox(root)
        finally:
            temporary.cleanup()


class FilesystemToolTests(ToolboxTestCase):
    def test_write_file_creates_parent_and_read_file_slices_inclusive_lines(self) -> None:
        written = self.toolbox.write_file("nested/example.txt", "one\ntwo\nthree\n")
        self.assertTrue(written.success)
        result = self.toolbox.read_file("nested/example.txt", 2, 3)
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "two\nthree\n")
        self.assertEqual(result.metadata["line_count"], 3)

    def test_read_file_rejects_absolute_and_traversing_paths(self) -> None:
        self.assertFalse(self.toolbox.read_file("../outside.txt").success)
        self.assertFalse(self.toolbox.read_file(str(self.root / "x.txt")).success)

    def test_write_file_rejects_path_that_escapes_through_symlink(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        result = self.toolbox.write_file("link/escaped.txt", "unsafe")
        self.assertFalse(result.success)
        self.assertFalse((outside / "escaped.txt").exists())

    def test_read_file_validates_line_ranges(self) -> None:
        self.toolbox.write_file("lines.txt", "a\nb\n")
        self.assertFalse(self.toolbox.read_file("lines.txt", 0).success)
        self.assertFalse(self.toolbox.read_file("lines.txt", 3, 2).success)


class GrepToolTests(ToolboxTestCase):
    def test_grep_searches_file_and_recursively_searches_directory(self) -> None:
        self.toolbox.write_file("src/a.py", "alpha\nmatch here\n")
        self.toolbox.write_file("src/nested/b.py", "another match\n")
        single = self.toolbox.grep("^match", "src/a.py")
        self.assertTrue(single.success)
        self.assertEqual(single.stdout, "src/a.py:2:match here")
        recursive = self.toolbox.grep("match", "src")
        self.assertTrue(recursive.success)
        self.assertEqual(recursive.metadata["matches"], 2)
        self.assertIn("src/nested/b.py:1:another match", recursive.stdout)

    def test_grep_reports_invalid_patterns_and_limits_matches(self) -> None:
        self.toolbox.write_file("many.txt", "hit\nhit\nhit\n")
        self.assertFalse(self.toolbox.grep("[", "many.txt").success)
        from tools.grep import grep

        result = grep(self.toolbox.workspace, "hit", "many.txt", max_matches=2)
        self.assertTrue(result.success)
        self.assertTrue(result.metadata["truncated"])
        self.assertEqual(result.metadata["matches"], 2)

    def test_grep_rejects_path_escape(self) -> None:
        self.assertFalse(self.toolbox.grep(".*", "../").success)


class ShellToolTests(ToolboxTestCase):
    def test_shell_runs_from_workspace_root_and_logs_command(self) -> None:
        result = self.toolbox.shell("pwd")
        self.assertTrue(result.success)
        self.assertEqual(Path(result.stdout.strip()), self.root.resolve())
        records = [json.loads(line) for line in self.toolbox.workspace.log_path.read_text().splitlines()]
        self.assertEqual(records[-1]["tool"], "shell")
        self.assertEqual(records[-1]["command"], "pwd")
        self.assertIn("timestamp", records[-1])

    def test_shell_filters_inherited_environment_and_redirects_home(self) -> None:
        with patch.dict(os.environ, {"TERMINAL_TEST_SECRET": "should-not-leak"}):
            result = self.toolbox.shell('printf "%s|%s" "$HOME" "${TERMINAL_TEST_SECRET-unset}"')
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, f"{self.root.resolve()}|unset")

    def test_shell_enforces_output_limit(self) -> None:
        limited = Toolbox(self.root, output_limit_bytes=8)
        try:
            result = limited.shell("printf '0123456789abcdef'")
        finally:
            limited.close()
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "01234567")
        self.assertTrue(result.stdout_truncated)

    def test_shell_times_out_and_kills_process(self) -> None:
        result = self.toolbox.shell("sleep 5", timeout=0.05)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error or "")

    def test_shell_rejects_obvious_destructive_or_workspace_escaping_commands(self) -> None:
        for command in (
            "rm -rf ../outside",
            "cd .. && pwd",
            "pushd /tmp",
            "sudo echo unsafe",
            "git reset --hard",
            "find . -delete",
        ):
            with self.subTest(command=command):
                result = self.toolbox.shell(command)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)

    def test_shell_validates_timeout(self) -> None:
        result = self.toolbox.shell("true", timeout=301)
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error or "")


class GitToolTests(ToolboxTestCase):
    def setUp(self) -> None:
        super().setUp()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)

    def test_git_operates_in_workspace_and_reads_status(self) -> None:
        self.toolbox.write_file("new.txt", "content\n")
        result = self.toolbox.git("status --short")
        self.assertTrue(result.success, result.stderr)
        self.assertIn("new.txt", result.stdout)

    def test_git_can_stage_workspace_file(self) -> None:
        self.toolbox.write_file("new.txt", "content\n")
        added = self.toolbox.git("add new.txt")
        self.assertTrue(added.success, added.stderr)
        status = self.toolbox.git("status --short")
        self.assertIn("A  new.txt", status.stdout)

    def test_git_blocks_host_or_destructive_operations(self) -> None:
        for command in (
            "reset --hard",
            "clean -fd",
            "push origin main",
            "status -C /tmp",
            "add ../outside.txt",
            "config --global user.name nobody",
        ):
            with self.subTest(command=command):
                result = self.toolbox.git(command)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)

    def test_git_does_not_use_shell_for_arguments(self) -> None:
        marker = self.root / "injected"
        result = self.toolbox.git(f"status; touch {marker}")
        self.assertFalse(result.success)
        self.assertFalse(marker.exists())

    def test_git_does_not_discover_a_repository_above_workspace(self) -> None:
        parent_repo = Path(self.temporary.name) / "parent-repository"
        nested_workspace = parent_repo / "task"
        nested_workspace.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=parent_repo, check=True)
        nested_toolbox = Toolbox(nested_workspace)
        try:
            result = nested_toolbox.git("status")
        finally:
            nested_toolbox.close()
        self.assertFalse(result.success)
        self.assertIn("not a git repository", result.stderr)

    def test_git_rejects_git_metadata_outside_workspace(self) -> None:
        external_repo = Path(self.temporary.name) / "external-repository"
        external_repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=external_repo, check=True)
        workspace = Path(self.temporary.name) / "linked-workspace"
        workspace.mkdir()
        (workspace / ".git").write_text(f"gitdir: {external_repo / '.git'}\n", encoding="utf-8")
        linked_toolbox = Toolbox(workspace)
        try:
            result = linked_toolbox.git("status")
        finally:
            linked_toolbox.close()
        self.assertFalse(result.success)
        self.assertIn("outside the workspace", result.error or "")


if __name__ == "__main__":
    unittest.main()
