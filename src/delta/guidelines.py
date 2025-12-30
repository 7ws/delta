"""AGENTS.md parser and guideline extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Guideline:
    """A single guideline from AGENTS.md."""

    id: str
    text: str


@dataclass
class MinorSection:
    """A minor section containing guidelines."""

    id: str
    name: str
    guidelines: list[Guideline] = field(default_factory=list)


@dataclass
class MajorSection:
    """A major section from AGENTS.md."""

    number: int
    name: str
    minor_sections: list[MinorSection] = field(default_factory=list)
    guidelines: list[Guideline] = field(default_factory=list)

    def all_guidelines(self) -> list[Guideline]:
        """Return all guidelines in this major section, including minor sections."""
        result = list(self.guidelines)
        for minor in self.minor_sections:
            result.extend(minor.guidelines)
        return result


@dataclass
class AgentsDocument:
    """Parsed AGENTS.md document."""

    raw_content: str
    major_sections: list[MajorSection] = field(default_factory=list)
    _section_contents: dict[int, str] = field(default_factory=dict)

    def get_section_names(self) -> list[str]:
        """Return list of major section names with numbers."""
        return [f"§{s.number} {s.name}" for s in self.major_sections]

    def get_section_content(self, section_number: int) -> str:
        """Return the raw content of a major section.

        Args:
            section_number: The section number to retrieve.

        Returns:
            Raw content of the section, or empty string if not found.
        """
        return self._section_contents.get(section_number, "")


def parse_agents_md(path: Path) -> AgentsDocument:
    """Parse AGENTS.md file and extract structure.

    Args:
        path: Path to AGENTS.md file.

    Returns:
        Parsed document with sections and guidelines.
    """
    content = path.read_text()
    doc = AgentsDocument(raw_content=content)

    major_pattern = re.compile(r"^# (\d+)\. (.+)$", re.MULTILINE)
    minor_pattern = re.compile(r"^## (\d+\.\d+) (.+)$", re.MULTILINE)
    guideline_pattern = re.compile(r"^- (\d+\.\d+\.\d+): (.+)$", re.MULTILINE)

    major_matches = list(major_pattern.finditer(content))

    for i, match in enumerate(major_matches):
        section_start = match.start()  # Include the header
        section_end = major_matches[i + 1].start() if i + 1 < len(major_matches) else len(content)
        full_section_content = content[section_start:section_end]
        section_content = content[match.end():section_end]  # Content after header

        section_number = int(match.group(1))
        major = MajorSection(number=section_number, name=match.group(2))

        # Store raw section content for parallel compliance reviews
        doc._section_contents[section_number] = full_section_content.strip()

        minor_matches = list(minor_pattern.finditer(section_content))

        if minor_matches:
            pre_minor_content = section_content[: minor_matches[0].start()]
            for g_match in guideline_pattern.finditer(pre_minor_content):
                major.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))

            for j, m_match in enumerate(minor_matches):
                minor_start = m_match.end()
                minor_end = (
                    minor_matches[j + 1].start()
                    if j + 1 < len(minor_matches)
                    else len(section_content)
                )
                minor_content = section_content[minor_start:minor_end]

                minor = MinorSection(id=m_match.group(1), name=m_match.group(2))
                for g_match in guideline_pattern.finditer(minor_content):
                    minor.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))
                major.minor_sections.append(minor)
        else:
            for g_match in guideline_pattern.finditer(section_content):
                major.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))

        doc.major_sections.append(major)

    return doc


def find_agents_md(start_path: Path | None = None) -> Path | None:
    """Find AGENTS.md file by searching upward from start_path.

    Args:
        start_path: Starting directory. Defaults to current working directory.

    Returns:
        Path to AGENTS.md if found, None otherwise.
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while current != current.parent:
        agents_path = current / "AGENTS.md"
        if agents_path.exists():
            return agents_path
        current = current.parent

    root_agents = current / "AGENTS.md"
    if root_agents.exists():
        return root_agents

    return None


def get_bundled_agents_md() -> Path:
    """Get the path to the bundled AGENTS.md shipped with Delta.

    Returns:
        Path to the bundled AGENTS.md file.

    Raises:
        FileNotFoundError: If the bundled file is missing (installation issue).
    """
    # Get the directory where this module is installed
    module_dir = Path(__file__).parent
    bundled_path = module_dir / "AGENTS.md"

    if bundled_path.exists():
        return bundled_path

    raise FileNotFoundError(
        "Bundled AGENTS.md not found. This indicates a Delta installation issue."
    )


def merge_agents_content(bundled_content: str, project_content: str) -> str:
    """Merge bundled AGENTS.md with project-specific overrides.

    The project AGENTS.md content is appended after the bundled content,
    with a clear separator. This allows project-specific guidelines to
    override or extend the bundled defaults.

    Args:
        bundled_content: Content from Delta's bundled AGENTS.md.
        project_content: Content from the project's local AGENTS.md.

    Returns:
        Merged content with project overrides appended.
    """
    separator = (
        "\n\n---\n\n"
        "# Project-Specific Guidelines\n\n"
        "The following guidelines are from the project's local AGENTS.md "
        "and take precedence over the defaults above.\n\n"
    )
    return bundled_content + separator + project_content


def load_merged_agents_md(cwd: Path | None = None) -> tuple[Path, str]:
    """Load AGENTS.md with bundled defaults and project overrides merged.

    Always loads the bundled AGENTS.md first, then merges with project-local
    AGENTS.md if one exists. Project guidelines take precedence.

    Args:
        cwd: Working directory to search for project AGENTS.md.

    Returns:
        Tuple of (path used for identification, merged content).
        Path is the project AGENTS.md if found, otherwise bundled path.
    """
    bundled_path = get_bundled_agents_md()
    bundled_content = bundled_path.read_text()

    project_path = find_agents_md(cwd)

    if project_path is not None and project_path != bundled_path:
        project_content = project_path.read_text()
        merged_content = merge_agents_content(bundled_content, project_content)
        return project_path, merged_content

    return bundled_path, bundled_content


def parse_agents_md_from_content(content: str) -> AgentsDocument:
    """Parse AGENTS.md content and extract structure.

    Args:
        content: Raw content of AGENTS.md file.

    Returns:
        Parsed document with sections and guidelines.
    """
    doc = AgentsDocument(raw_content=content)

    major_pattern = re.compile(r"^# (\d+)\. (.+)$", re.MULTILINE)
    minor_pattern = re.compile(r"^## (\d+\.\d+) (.+)$", re.MULTILINE)
    guideline_pattern = re.compile(r"^- (\d+\.\d+\.\d+): (.+)$", re.MULTILINE)

    major_matches = list(major_pattern.finditer(content))

    for i, match in enumerate(major_matches):
        section_start = match.start()  # Include the header
        section_end = major_matches[i + 1].start() if i + 1 < len(major_matches) else len(content)
        full_section_content = content[section_start:section_end]
        section_content = content[match.end():section_end]  # Content after header

        section_number = int(match.group(1))
        major = MajorSection(number=section_number, name=match.group(2))

        # Store raw section content for parallel compliance reviews
        doc._section_contents[section_number] = full_section_content.strip()

        minor_matches = list(minor_pattern.finditer(section_content))

        if minor_matches:
            pre_minor_content = section_content[: minor_matches[0].start()]
            for g_match in guideline_pattern.finditer(pre_minor_content):
                major.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))

            for j, m_match in enumerate(minor_matches):
                minor_start = m_match.end()
                minor_end = (
                    minor_matches[j + 1].start()
                    if j + 1 < len(minor_matches)
                    else len(section_content)
                )
                minor_content = section_content[minor_start:minor_end]

                minor = MinorSection(id=m_match.group(1), name=m_match.group(2))
                for g_match in guideline_pattern.finditer(minor_content):
                    minor.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))
                major.minor_sections.append(minor)
        else:
            for g_match in guideline_pattern.finditer(section_content):
                major.guidelines.append(Guideline(id=g_match.group(1), text=g_match.group(2)))

        doc.major_sections.append(major)

    return doc
