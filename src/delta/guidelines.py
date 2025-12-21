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

    def get_section_names(self) -> list[str]:
        """Return list of major section names with numbers."""
        return [f"§{s.number} {s.name}" for s in self.major_sections]


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
        section_start = match.end()
        section_end = major_matches[i + 1].start() if i + 1 < len(major_matches) else len(content)
        section_content = content[section_start:section_end]

        major = MajorSection(number=int(match.group(1)), name=match.group(2))

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
