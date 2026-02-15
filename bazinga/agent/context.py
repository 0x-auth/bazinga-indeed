"""
BAZINGA Agent Auto-Context - Understands your project automatically.

"The first AI you actually own"

When you start the agent in a directory, this module:
1. Detects project type (Python, JS, Go, Rust, etc.)
2. Reads key configuration files
3. Maps the project structure
4. Provides this context to every query
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ProjectContext:
    """Context about the current project."""

    root: Path
    project_type: str = "unknown"
    name: str = ""
    description: str = ""

    # Key files content
    readme_summary: str = ""
    config_summary: str = ""

    # Structure
    directories: List[str] = field(default_factory=list)
    key_files: List[str] = field(default_factory=list)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)

    # Git info
    git_branch: str = ""
    git_status: str = ""

    def to_prompt(self) -> str:
        """Convert context to prompt string for LLM."""
        lines = ["## Project Context (auto-detected):"]

        if self.name:
            lines.append(f"Project: {self.name}")
        lines.append(f"Type: {self.project_type}")
        lines.append(f"Root: {self.root}")

        if self.git_branch:
            lines.append(f"Git: {self.git_branch}")

        if self.description:
            lines.append(f"\nDescription: {self.description[:200]}")

        if self.key_files:
            lines.append(f"\nKey files: {', '.join(self.key_files[:10])}")

        if self.directories:
            lines.append(f"Directories: {', '.join(self.directories[:10])}")

        if self.dependencies:
            lines.append(f"Dependencies: {', '.join(self.dependencies[:10])}")

        if self.readme_summary:
            lines.append(f"\nREADME summary: {self.readme_summary[:300]}")

        return "\n".join(lines)


class ProjectDetector:
    """Detects project type and extracts context."""

    # Project type indicators
    PROJECT_INDICATORS = {
        "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
        "javascript": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
        "typescript": ["tsconfig.json", "package.json"],
        "rust": ["Cargo.toml"],
        "go": ["go.mod", "go.sum"],
        "ruby": ["Gemfile", "*.gemspec"],
        "java": ["pom.xml", "build.gradle"],
        "csharp": ["*.csproj", "*.sln"],
        "php": ["composer.json"],
        "swift": ["Package.swift", "*.xcodeproj"],
    }

    # Key files to read for context
    KEY_FILES = [
        "README.md", "README", "readme.md",
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
        ".env.example", "docker-compose.yml", "Makefile",
    ]

    # Directories to note
    IMPORTANT_DIRS = [
        "src", "lib", "app", "pkg", "cmd",
        "tests", "test", "spec",
        "docs", "documentation",
        "scripts", "bin",
        "config", "conf",
    ]

    def __init__(self, root: Path = None):
        self.root = root or Path.cwd()

    def detect(self) -> ProjectContext:
        """Detect project type and build context."""
        ctx = ProjectContext(root=self.root)

        # Detect project type
        ctx.project_type = self._detect_type()

        # Get project name and description
        ctx.name, ctx.description = self._get_project_info()

        # Find key files
        ctx.key_files = self._find_key_files()

        # Find important directories
        ctx.directories = self._find_directories()

        # Get dependencies
        ctx.dependencies = self._get_dependencies()

        # Get README summary
        ctx.readme_summary = self._get_readme_summary()

        # Get git info
        ctx.git_branch, ctx.git_status = self._get_git_info()

        return ctx

    def _detect_type(self) -> str:
        """Detect project type based on files present."""
        for proj_type, indicators in self.PROJECT_INDICATORS.items():
            for indicator in indicators:
                if "*" in indicator:
                    # Glob pattern
                    if list(self.root.glob(indicator)):
                        return proj_type
                else:
                    if (self.root / indicator).exists():
                        return proj_type
        return "unknown"

    def _get_project_info(self) -> tuple:
        """Extract project name and description from config files."""
        name = self.root.name
        description = ""

        # Try pyproject.toml
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                # Simple parsing (avoid toml dependency)
                for line in content.split('\n'):
                    if line.startswith('name = '):
                        name = line.split('=')[1].strip().strip('"\'')
                    if line.startswith('description = '):
                        description = line.split('=', 1)[1].strip().strip('"\'')
            except:
                pass

        # Try package.json
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                name = data.get("name", name)
                description = data.get("description", description)
            except:
                pass

        # Try Cargo.toml
        cargo = self.root / "Cargo.toml"
        if cargo.exists():
            try:
                content = cargo.read_text()
                for line in content.split('\n'):
                    if line.startswith('name = '):
                        name = line.split('=')[1].strip().strip('"\'')
                    if line.startswith('description = '):
                        description = line.split('=', 1)[1].strip().strip('"\'')
            except:
                pass

        return name, description

    def _find_key_files(self) -> List[str]:
        """Find key configuration and documentation files."""
        found = []
        for filename in self.KEY_FILES:
            path = self.root / filename
            if path.exists():
                found.append(filename)
        return found

    def _find_directories(self) -> List[str]:
        """Find important project directories."""
        found = []
        for dirname in self.IMPORTANT_DIRS:
            path = self.root / dirname
            if path.exists() and path.is_dir():
                found.append(dirname)

        # Also add any top-level directories that look important
        for item in self.root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name not in found and item.name not in ['node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build']:
                    found.append(item.name)

        return found[:15]  # Limit

    def _get_dependencies(self) -> List[str]:
        """Extract main dependencies."""
        deps = []

        # Python
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                in_deps = False
                for line in content.split('\n'):
                    if 'dependencies' in line and '=' in line:
                        in_deps = True
                        continue
                    if in_deps:
                        if line.startswith('[') or line.strip() == ']':
                            in_deps = False
                        elif '"' in line:
                            dep = line.strip().strip('",')
                            if dep and not dep.startswith('#'):
                                # Extract package name
                                dep_name = dep.split('>=')[0].split('==')[0].split('<')[0].strip()
                                if dep_name:
                                    deps.append(dep_name)
            except:
                pass

        # Node.js
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                deps.extend(list(data.get("dependencies", {}).keys())[:10])
            except:
                pass

        return deps[:15]

    def _get_readme_summary(self) -> str:
        """Get first paragraph of README."""
        for name in ["README.md", "README", "readme.md"]:
            readme = self.root / name
            if readme.exists():
                try:
                    content = readme.read_text()
                    # Get first meaningful paragraph
                    lines = content.split('\n')
                    summary_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('```'):
                            summary_lines.append(line)
                            if len(' '.join(summary_lines)) > 200:
                                break
                    return ' '.join(summary_lines)[:300]
                except:
                    pass
        return ""

    def _get_git_info(self) -> tuple:
        """Get current git branch and status."""
        branch = ""
        status = ""

        git_head = self.root / ".git" / "HEAD"
        if git_head.exists():
            try:
                content = git_head.read_text().strip()
                if content.startswith("ref: refs/heads/"):
                    branch = content.replace("ref: refs/heads/", "")
            except:
                pass

        return branch, status


def get_project_context(root: Path = None) -> ProjectContext:
    """Get context for current project."""
    detector = ProjectDetector(root)
    return detector.detect()


def get_project_prompt(root: Path = None) -> str:
    """Get project context as prompt string."""
    ctx = get_project_context(root)
    return ctx.to_prompt()
