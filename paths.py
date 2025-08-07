"""
Centralized path management for the data analysis agent.

This module provides a single source of truth for all paths used throughout the application,
ensuring consistency and preventing directory-related issues.
"""

from pathlib import Path
from typing import Optional


class ProjectPaths:
    """
    Centralized path management for the data analysis agent.

    All paths are resolved relative to the project root directory,
    ensuring consistency regardless of where the script is run from.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize project paths.

        Args:
            project_root: Explicit project root path. If None, auto-detects based on this file's location.
        """
        if project_root is None:
            # Auto-detect project root - this file is in the project root
            self._project_root = Path(__file__).parent.absolute()
        else:
            self._project_root = Path(project_root).absolute()

        # Validate project root
        if not self._project_root.exists():
            raise ValueError(f"Project root does not exist: {self._project_root}")

    @property
    def project_root(self) -> Path:
        """The absolute path to the project root directory."""
        return self._project_root

    @property
    def data_dir(self) -> Path:
        """The main data directory."""
        return self._project_root / "data"

    @property
    def plan_dir(self) -> Path:
        """Directory for storing analysis plans."""
        return self.data_dir / "plan"

    @property
    def code_dir(self) -> Path:
        """Directory for storing generated code."""
        return self.data_dir / "code"

    @property
    def tables_dir(self) -> Path:
        """Directory containing input data tables."""
        return self.data_dir / "InfiAgent-DABench" / "da-dev-tables"

    @property
    def log_file(self) -> Path:
        """Main application log file."""
        return self._project_root / "data_analysis_agent.log"

    @property
    def merged_plans_file(self) -> Path:
        """Output file for merged plans."""
        return self.data_dir / "merged_with_plans.csv"

    @property
    def merged_code_file(self) -> Path:
        """Output file for merged code."""
        return self.data_dir / "merged_with_code.csv"

    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.plan_dir,
            self.code_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def clean_output_directories(self, skip_log_files: bool = True):
        """
        Clean output directories, removing old files.

        Args:
            skip_log_files: If True, skip files ending with .log to avoid file locking issues.
        """
        directories_to_clean = [self.plan_dir, self.code_dir]
        files_to_clean = [self.merged_plans_file, self.merged_code_file]

        files_removed = 0

        # Clean directories
        for directory in directories_to_clean:
            if directory.exists():
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        # Skip log files to avoid file locking issues
                        if skip_log_files and file_path.name.endswith(".log"):
                            continue
                        try:
                            file_path.unlink()
                            files_removed += 1
                        except OSError:
                            # Silently ignore file deletion errors
                            pass
                    elif file_path.is_dir():
                        try:
                            import shutil

                            shutil.rmtree(file_path)
                            files_removed += 1
                        except OSError:
                            # Silently ignore directory deletion errors
                            pass

        # Clean old output files
        for file_path in files_to_clean:
            if file_path.exists():
                try:
                    file_path.unlink()
                    files_removed += 1
                except OSError:
                    # Silently ignore file deletion errors
                    pass

        return files_removed

    def get_data_file_path(self, filename: str) -> Path:
        """
        Get the full path to a data file in the tables directory.

        Args:
            filename: The name of the data file.

        Returns:
            Full path to the data file.
        """
        return self.tables_dir / filename

    def __str__(self) -> str:
        return f"ProjectPaths(root={self.project_root})"

    def __repr__(self) -> str:
        return self.__str__()


class PathsManager:
    """
    Singleton manager for ProjectPaths instances.

    This provides a cleaner alternative to global variables while maintaining
    easy access patterns throughout the application.
    """

    _instance: Optional["ProjectPaths"] = None
    _project_root: Optional[Path] = None

    @classmethod
    def get_instance(cls, project_root: Optional[Path] = None) -> "ProjectPaths":
        """
        Get or create the ProjectPaths instance.

        Args:
            project_root: Optional project root path. Only used on first call.

        Returns:
            The ProjectPaths instance.
        """
        if cls._instance is None or (
            project_root is not None and cls._project_root != project_root
        ):
            cls._project_root = project_root
            cls._instance = ProjectPaths(project_root)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._project_root = None


def get_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """
    Get the ProjectPaths instance.

    Args:
        project_root: Optional project root path. Only used on first call.

    Returns:
        The ProjectPaths instance.
    """
    return PathsManager.get_instance(project_root)


def set_project_root(root_path: Path) -> ProjectPaths:
    """
    Set a custom project root path and return the new instance.

    Args:
        root_path: New project root path.

    Returns:
        New ProjectPaths instance with the specified root.
    """
    return PathsManager.get_instance(root_path)


# Convenience instance for backward compatibility
# This maintains the same import pattern: from paths import paths
paths = get_paths()
