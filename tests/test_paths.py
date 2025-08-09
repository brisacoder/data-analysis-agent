"""
Unit tests for the paths module.

These tests validate the path management functionality including
ProjectPaths class, PathsManager singleton, and related utilities.
"""

import tempfile
import pytest
import platform
from pathlib import Path

from data_analysis_agent.paths import (
    ProjectPaths,
    PathsManager,
    get_paths,
    set_project_root,
    paths,
)


class TestProjectPaths:
    """Test the ProjectPaths class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def project_paths(self, temp_dir):
        """Create a ProjectPaths instance with a temporary directory."""
        return ProjectPaths(temp_dir)

    def test_initialization_with_explicit_root(self, temp_dir):
        """Test ProjectPaths initialization with explicit project root."""
        project_paths = ProjectPaths(temp_dir)
        assert project_paths.project_root == temp_dir.absolute()

    def test_initialization_with_nonexistent_root(self):
        """Test ProjectPaths initialization with non-existent root raises ValueError."""
        nonexistent_path = Path("/nonexistent/path/that/should/not/exist")
        with pytest.raises(ValueError, match="Project root does not exist"):
            ProjectPaths(nonexistent_path)

    def test_auto_detection_of_project_root(self):
        """Test that ProjectPaths can auto-detect project root."""
        # This should work without raising an exception
        project_paths = ProjectPaths()
        assert project_paths.project_root.exists()
        assert project_paths.project_root.is_dir()

    def test_project_root_property(self, project_paths, temp_dir):
        """Test the project_root property."""
        assert project_paths.project_root == temp_dir.absolute()

    def test_data_dir_property(self, project_paths, temp_dir):
        """Test the data_dir property."""
        expected_path = temp_dir / "data"
        assert project_paths.data_dir == expected_path

    def test_plan_dir_property(self, project_paths, temp_dir):
        """Test the plan_dir property."""
        expected_path = temp_dir / "data" / "plan"
        assert project_paths.plan_dir == expected_path

    def test_code_dir_property(self, project_paths, temp_dir):
        """Test the code_dir property."""
        expected_path = temp_dir / "data" / "code"
        assert project_paths.code_dir == expected_path

    def test_tables_dir_property(self, project_paths, temp_dir):
        """Test the tables_dir property."""
        expected_path = temp_dir / "data" / "InfiAgent-DABench" / "da-dev-tables"
        assert project_paths.tables_dir == expected_path

    def test_log_file_property(self, project_paths, temp_dir):
        """Test the log_file property."""
        expected_path = temp_dir / "data" / "data_analysis_agent.log"
        assert project_paths.log_file == expected_path

    def test_merged_plans_file_property(self, project_paths, temp_dir):
        """Test the merged_plans_file property."""
        expected_path = temp_dir / "data" / "merged_with_plans.csv"
        assert project_paths.merged_plans_file == expected_path

    def test_merged_code_file_property(self, project_paths, temp_dir):
        """Test the merged_code_file property."""
        expected_path = temp_dir / "data" / "merged_with_code.csv"
        assert project_paths.merged_code_file == expected_path

    def test_ensure_directories(self, project_paths):
        """Test that ensure_directories creates necessary directories."""
        # Initially, directories should not exist
        assert not project_paths.data_dir.exists()
        assert not project_paths.plan_dir.exists()
        assert not project_paths.code_dir.exists()

        # After calling ensure_directories, they should exist
        project_paths.ensure_directories()
        assert project_paths.data_dir.exists()
        assert project_paths.plan_dir.exists()
        assert project_paths.code_dir.exists()

        # Calling again should not raise an error
        project_paths.ensure_directories()

    def test_get_data_file_path(self, project_paths):
        """Test the get_data_file_path method."""
        filename = "test_data.csv"
        expected_path = project_paths.tables_dir / filename
        assert project_paths.get_data_file_path(filename) == expected_path

    def test_clean_output_directories_empty(self, project_paths):
        """Test cleaning empty directories."""
        # Create directories first
        project_paths.ensure_directories()

        # Clean should return 0 for empty directories
        files_removed = project_paths.clean_output_directories()
        assert files_removed == 0

    def test_clean_output_directories_with_files(self, project_paths):
        """Test cleaning directories with files."""
        # Create directories and add some files
        project_paths.ensure_directories()

        # Add files to plan_dir
        test_file1 = project_paths.plan_dir / "test_plan.json"
        test_file1.write_text("test content")

        # Add files to code_dir
        test_file2 = project_paths.code_dir / "test_code.py"
        test_file2.write_text("print('test')")

        # Add log file that should be skipped
        log_file = project_paths.plan_dir / "test.log"
        log_file.write_text("log content")

        # Add merged files
        project_paths.merged_plans_file.write_text("plans")
        project_paths.merged_code_file.write_text("code")

        # Clean directories (skip log files by default)
        files_removed = project_paths.clean_output_directories(skip_log_files=True)

        # Should remove all files except log files
        assert files_removed == 4  # 2 regular files + 2 merged files
        assert not test_file1.exists()
        assert not test_file2.exists()
        assert log_file.exists()  # Should still exist
        assert not project_paths.merged_plans_file.exists()
        assert not project_paths.merged_code_file.exists()

    def test_clean_output_directories_including_logs(self, project_paths):
        """Test cleaning directories including log files."""
        # Create directories and add files
        project_paths.ensure_directories()

        log_file = project_paths.plan_dir / "test.log"
        log_file.write_text("log content")

        # Clean without skipping log files
        files_removed = project_paths.clean_output_directories(skip_log_files=False)

        assert files_removed == 1
        assert not log_file.exists()

    def test_clean_output_directories_with_subdirectories(self, project_paths):
        """Test cleaning directories with subdirectories."""
        # Create directories and subdirectories
        project_paths.ensure_directories()

        subdir = project_paths.plan_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        files_removed = project_paths.clean_output_directories()

        assert files_removed == 1  # Should count the subdirectory as one item
        assert not subdir.exists()

    def test_clean_output_directories_handles_errors(self, project_paths, monkeypatch):
        """Test that clean_output_directories handles file deletion errors gracefully."""
        project_paths.ensure_directories()

        test_file = project_paths.plan_dir / "test.txt"
        test_file.write_text("content")

        # Mock unlink to raise OSError
        original_unlink = Path.unlink

        def mock_unlink(self, missing_ok=False):
            if self.name == "test.txt":
                raise OSError("Permission denied")
            return original_unlink(self, missing_ok)

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        # Should not raise an exception and return 0 files removed
        files_removed = project_paths.clean_output_directories()
        assert files_removed == 0

    def test_str_representation(self, project_paths, temp_dir):
        """Test string representation of ProjectPaths."""
        expected_str = f"ProjectPaths(root={temp_dir.absolute()})"
        assert str(project_paths) == expected_str

    def test_repr_representation(self, project_paths, temp_dir):
        """Test repr representation of ProjectPaths."""
        expected_repr = f"ProjectPaths(root={temp_dir.absolute()})"
        assert repr(project_paths) == expected_repr


class TestPathsManager:
    """Test the PathsManager singleton class."""

    def setup_method(self):
        """Reset the singleton before each test."""
        PathsManager.reset()

    def teardown_method(self):
        """Reset the singleton after each test."""
        PathsManager.reset()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_get_instance_creates_instance(self, temp_dir):
        """Test that get_instance creates a ProjectPaths instance."""
        instance = PathsManager.get_instance(temp_dir)
        assert isinstance(instance, ProjectPaths)
        assert instance.project_root == temp_dir.absolute()

    def test_get_instance_returns_same_instance(self, temp_dir):
        """Test that multiple calls to get_instance return the same instance."""
        instance1 = PathsManager.get_instance(temp_dir)
        instance2 = PathsManager.get_instance()
        assert instance1 is instance2

    def test_get_instance_with_different_root_creates_new_instance(self, temp_dir):
        """Test that providing a different root creates a new instance."""
        instance1 = PathsManager.get_instance(temp_dir)

        with tempfile.TemporaryDirectory() as temp_dir2:
            temp_dir2_path = Path(temp_dir2)
            instance2 = PathsManager.get_instance(temp_dir2_path)

            assert instance1 is not instance2
            assert instance2.project_root == temp_dir2_path.absolute()

    def test_reset_clears_instance(self, temp_dir):
        """Test that reset clears the singleton instance."""
        instance1 = PathsManager.get_instance(temp_dir)
        PathsManager.reset()
        instance2 = PathsManager.get_instance(temp_dir)

        assert instance1 is not instance2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Reset the singleton before each test."""
        PathsManager.reset()

    def teardown_method(self):
        """Reset the singleton after each test."""
        PathsManager.reset()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_get_paths_returns_project_paths_instance(self, temp_dir):
        """Test that get_paths returns a ProjectPaths instance."""
        instance = get_paths(temp_dir)
        assert isinstance(instance, ProjectPaths)
        assert instance.project_root == temp_dir.absolute()

    def test_get_paths_without_root_uses_auto_detection(self):
        """Test that get_paths without root auto-detects project root."""
        instance = get_paths()
        assert isinstance(instance, ProjectPaths)
        assert instance.project_root.exists()

    def test_set_project_root_creates_new_instance(self, temp_dir):
        """Test that set_project_root creates a new instance with the specified root."""
        instance = set_project_root(temp_dir)
        assert isinstance(instance, ProjectPaths)
        assert instance.project_root == temp_dir.absolute()

    def test_paths_convenience_instance_exists(self):
        """Test that the paths convenience instance is available."""
        # The paths instance should be a ProjectPaths instance
        assert isinstance(paths, ProjectPaths)


class TestIntegration:
    """Integration tests for the paths module."""

    def setup_method(self):
        """Reset the singleton before each test."""
        PathsManager.reset()

    def teardown_method(self):
        """Reset the singleton after each test."""
        PathsManager.reset()

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a realistic project structure
            (temp_path / "data").mkdir()
            (temp_path / "data" / "plan").mkdir()
            (temp_path / "data" / "code").mkdir()
            (temp_path / "data" / "InfiAgent-DABench").mkdir()
            (temp_path / "data" / "InfiAgent-DABench" / "da-dev-tables").mkdir()

            # Add some test files
            (temp_path / "data" / "plan" / "old_plan.json").write_text('{"test": "data"}')
            (temp_path / "data" / "code" / "old_code.py").write_text("print('old')")
            (temp_path / "data" / "InfiAgent-DABench" / "da-dev-tables" / "test.csv").write_text("col1,col2\n1,2")

            yield temp_path

    def test_full_workflow(self, temp_project):
        """Test a complete workflow using the paths module."""
        # Initialize paths
        project_paths = get_paths(temp_project)

        # Verify all paths are correct
        assert project_paths.project_root == temp_project.absolute()
        assert project_paths.data_dir == temp_project / "data"

        # Test file access
        data_file_path = project_paths.get_data_file_path("test.csv")
        assert data_file_path.exists()
        assert data_file_path.read_text() == "col1,col2\n1,2"

        # Test directory cleaning
        files_removed = project_paths.clean_output_directories()
        assert files_removed == 2  # old_plan.json and old_code.py

        # Verify data files are preserved
        assert data_file_path.exists()

        # Test directory recreation
        project_paths.ensure_directories()
        assert project_paths.plan_dir.exists()
        assert project_paths.code_dir.exists()

    def test_singleton_behavior_across_functions(self, temp_project):
        """Test that singleton behavior works across different access patterns."""
        # Get instance through different methods
        instance1 = get_paths(temp_project)
        instance2 = set_project_root(temp_project)
        instance3 = PathsManager.get_instance()

        # All should be the same instance
        assert instance1 is instance2
        assert instance2 is instance3

        # All should have the same project root
        assert instance1.project_root == temp_project.absolute()
        assert instance2.project_root == temp_project.absolute()
        assert instance3.project_root == temp_project.absolute()


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility features."""

    def setup_method(self):
        """Reset the singleton before each test."""
        PathsManager.reset()

    def teardown_method(self):
        """Reset the singleton after each test."""
        PathsManager.reset()

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a realistic project structure
            (temp_path / "data").mkdir()
            (temp_path / "data" / "plan").mkdir()
            (temp_path / "data" / "code").mkdir()
            (temp_path / "data" / "InfiAgent-DABench").mkdir()
            (temp_path / "data" / "InfiAgent-DABench" / "da-dev-tables").mkdir()

            yield temp_path

    def test_path_separators_are_handled_correctly(self, temp_project):
        """Test that path separators work correctly on all platforms."""
        project_paths = ProjectPaths(temp_project)
        
        # Test that all paths use pathlib and are absolute
        paths_to_test = [
            project_paths.project_root,
            project_paths.data_dir,
            project_paths.plan_dir,
            project_paths.code_dir,
            project_paths.tables_dir,
            project_paths.log_file,
            project_paths.merged_plans_file,
            project_paths.merged_code_file,
        ]
        
        for path in paths_to_test:
            assert isinstance(path, Path), f"Path {path} should be a Path object"
            assert path.is_absolute(), f"Path {path} should be absolute"
            
            # Test that path parts work correctly
            parts = path.parts
            assert len(parts) > 0, f"Path {path} should have parts"
            
            # Test string representation works
            path_str = str(path)
            assert len(path_str) > 0, f"Path {path} should have string representation"

    def test_directory_operations_work_cross_platform(self, temp_project):
        """Test that directory operations work on all platforms."""
        project_paths = ProjectPaths(temp_project)
        
        # Test directory creation
        project_paths.ensure_directories()
        
        directories_to_check = [
            project_paths.data_dir,
            project_paths.plan_dir,
            project_paths.code_dir,
        ]
        
        for directory in directories_to_check:
            assert directory.exists(), f"Directory {directory} should exist"
            assert directory.is_dir(), f"Path {directory} should be a directory"
            
            # Test that we can create files in the directory
            test_file = directory / "test_file.txt"
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"
            
            # Clean up
            test_file.unlink()

    def test_file_operations_work_cross_platform(self, temp_project):
        """Test that file operations work on all platforms."""
        project_paths = ProjectPaths(temp_project)
        project_paths.ensure_directories()
        
        # Test various file types and names
        test_files = [
            (project_paths.plan_dir / "plan.json", '{"test": "data"}'),
            (project_paths.code_dir / "code.py", "print('hello')"),
            (project_paths.plan_dir / "file with spaces.txt", "content"),
            (project_paths.code_dir / "file-with-dashes.py", "# comment"),
        ]
        
        for file_path, content in test_files:
            # Write file
            file_path.write_text(content)
            assert file_path.exists()
            assert file_path.read_text() == content
            
            # Test file properties
            assert file_path.is_file()
            assert file_path.parent.exists()
            
        # Test cleanup
        files_removed = project_paths.clean_output_directories()
        assert files_removed >= len(test_files)

    def test_path_resolution_works_cross_platform(self, temp_project):
        """Test that path resolution works correctly."""
        project_paths = ProjectPaths(temp_project)
        
        # Test get_data_file_path with various filename patterns
        test_filenames = [
            "simple.csv",
            "file with spaces.csv",
            "file-with-dashes.csv",
            "file_with_underscores.csv",
            "file.with.dots.csv",
        ]
        
        for filename in test_filenames:
            resolved_path = project_paths.get_data_file_path(filename)
            expected_path = project_paths.tables_dir / filename
            
            assert resolved_path == expected_path
            assert isinstance(resolved_path, Path)
            assert resolved_path.is_absolute()

    def test_auto_detection_works_cross_platform(self):
        """Test that auto-detection works on the current platform."""
        project_paths = ProjectPaths()
        
        # Should successfully auto-detect
        assert project_paths.project_root.exists()
        assert project_paths.project_root.is_dir()
        
        # Should be able to find expected project files
        expected_files = ["pyproject.toml", "README.md"]
        project_files = [f.name for f in project_paths.project_root.iterdir() if f.is_file()]
        
        for expected_file in expected_files:
            assert expected_file in project_files, f"Expected to find {expected_file} in project root"

    def test_platform_specific_path_features(self, temp_project):
        """Test platform-specific path features are handled correctly."""
        project_paths = ProjectPaths(temp_project)
        
        # Test absolute path properties
        abs_path = project_paths.data_dir.absolute()
        assert abs_path.is_absolute()
        
        # Test that paths work with platform-specific features
        import platform
        system = platform.system()
        
        if system == "Windows":
            # On Windows, absolute paths should have drive letters
            path_str = str(abs_path)
            assert len(path_str) > 2 and path_str[1] == ":", f"Windows path should have drive letter: {path_str}"
            
        elif system in ["Linux", "Darwin"]:
            # On Unix systems, absolute paths should start with /
            path_str = str(abs_path)
            assert path_str.startswith("/"), f"Unix path should start with /: {path_str}"
        
        # Test path parts work correctly regardless of platform
        parts = abs_path.parts
        assert len(parts) > 0
        assert "data" in parts

    def test_relative_path_handling(self, temp_project):
        """Test that relative paths are handled correctly."""
        project_paths = ProjectPaths(temp_project)
        
        # Test that all paths are resolved to absolute
        paths_to_check = [
            project_paths.data_dir,
            project_paths.plan_dir,
            project_paths.code_dir,
            project_paths.tables_dir,
        ]
        
        for path in paths_to_check:
            assert path.is_absolute(), f"Path {path} should be absolute"
            
            # Test relative_to works
            try:
                relative = path.relative_to(project_paths.project_root)
                # Should be able to reconstruct the absolute path
                reconstructed = project_paths.project_root / relative
                assert reconstructed == path
            except ValueError:
                # This is okay if paths are on different drives (Windows)
                pass
