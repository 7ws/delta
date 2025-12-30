"""Tests for AGENTS.md parsing"""

from pathlib import Path

import pytest

from delta.guidelines import (
    find_agents_md,
    load_merged_agents_md,
    merge_agents_content,
    parse_agents_md,
    parse_agents_md_from_content,
)


@pytest.fixture
def sample_agents_md(tmp_path: Path) -> Path:
    """Sample AGENTS.md file for testing"""
    content = """# 1. Writing Style

## 1.1 Voice and Tense

- 1.1.1: Use active voice.
- 1.1.2: Use simple present tense.

## 1.2 Sentence Structure

- 1.2.1: Keep sentences under 30 words.

# 2. Technical Conduct

- 2.0.1: Read code before proposing changes.
- 2.0.2: Search the codebase exhaustively.

# 3. Git Operations

## 3.1 Branch Management

- 3.1.1: Name branches as type/component/short-title.
"""
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text(content)
    return agents_path


class TestParseMajorSections:
    """Tests for major section parsing"""

    def test_given_agents_md_with_three_sections_when_parsed_then_returns_three_major_sections(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)

        # Then
        assert len(doc.major_sections) == 3

    def test_given_agents_md_when_parsed_then_extracts_section_numbers(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)

        # Then
        assert doc.major_sections[0].number == 1
        assert doc.major_sections[1].number == 2
        assert doc.major_sections[2].number == 3

    def test_given_agents_md_when_parsed_then_extracts_section_names(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)

        # Then
        assert doc.major_sections[0].name == "Writing Style"
        assert doc.major_sections[1].name == "Technical Conduct"
        assert doc.major_sections[2].name == "Git Operations"


class TestParseMinorSections:
    """Tests for minor section parsing"""

    def test_given_major_section_with_minor_sections_when_parsed_then_extracts_minor_sections(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)
        writing_style = doc.major_sections[0]

        # Then
        assert len(writing_style.minor_sections) == 2

    def test_given_minor_section_when_parsed_then_extracts_id_and_name(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)
        voice_section = doc.major_sections[0].minor_sections[0]

        # Then
        assert voice_section.id == "1.1"
        assert voice_section.name == "Voice and Tense"


class TestParseGuidelines:
    """Tests for guideline parsing"""

    def test_given_minor_section_with_guidelines_when_parsed_then_extracts_guidelines(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)
        voice_section = doc.major_sections[0].minor_sections[0]

        # Then
        assert len(voice_section.guidelines) == 2
        assert voice_section.guidelines[0].id == "1.1.1"
        assert voice_section.guidelines[0].text == "Use active voice."

    def test_given_major_section_without_minor_sections_when_parsed_then_extracts_direct_guidelines(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)
        technical_conduct = doc.major_sections[1]

        # Then
        assert len(technical_conduct.guidelines) == 2
        assert technical_conduct.guidelines[0].id == "2.0.1"

    def test_given_major_section_with_mixed_content_when_all_guidelines_called_then_returns_all(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        path = sample_agents_md

        # When
        doc = parse_agents_md(path)
        all_guidelines = doc.major_sections[0].all_guidelines()

        # Then
        assert len(all_guidelines) == 3


class TestGetSectionNames:
    """Tests for section name formatting"""

    def test_given_parsed_document_when_get_section_names_called_then_returns_formatted_names(
        self, sample_agents_md: Path
    ) -> None:
        # Given
        doc = parse_agents_md(sample_agents_md)

        # When
        names = doc.get_section_names()

        # Then
        assert names == [
            "§1 Writing Style",
            "§2 Technical Conduct",
            "§3 Git Operations",
        ]


class TestFindAgentsMd:
    """Tests for AGENTS.md file discovery"""

    def test_given_agents_md_in_current_dir_when_find_called_then_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        agents_path = tmp_path / "AGENTS.md"
        agents_path.write_text("# 1. Test")
        monkeypatch.chdir(tmp_path)

        # When
        found = find_agents_md()

        # Then
        assert found == agents_path

    def test_given_agents_md_in_parent_dir_when_find_called_from_subdir_then_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        agents_path = tmp_path / "AGENTS.md"
        agents_path.write_text("# 1. Test")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        # When
        found = find_agents_md()

        # Then
        assert found == agents_path

    def test_given_no_agents_md_when_find_called_then_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.chdir(tmp_path)

        # When
        found = find_agents_md()

        # Then
        assert found is None


class TestMergeAgentsContent:
    """Tests for merging bundled and project AGENTS.md content."""

    def test_given_both_contents_when_merged_then_bundled_comes_first(self) -> None:
        # Given
        bundled = "# 1. Bundled Guidelines\n- 1.0.1: Rule one."
        project = "# 1. Override\n- 1.0.1: Project rule."

        # When
        merged = merge_agents_content(bundled, project)

        # Then
        bundled_pos = merged.find("# 1. Bundled Guidelines")
        project_pos = merged.find("# 1. Override")
        assert bundled_pos < project_pos

    def test_given_both_contents_when_merged_then_contains_separator(self) -> None:
        # Given
        bundled = "# 1. Bundled"
        project = "# 2. Project"

        # When
        merged = merge_agents_content(bundled, project)

        # Then
        assert "Project-Specific Guidelines" in merged
        assert "take precedence" in merged

    def test_given_both_contents_when_merged_then_contains_both_contents(self) -> None:
        # Given
        bundled = "Bundled content here"
        project = "Project content here"

        # When
        merged = merge_agents_content(bundled, project)

        # Then
        assert "Bundled content here" in merged
        assert "Project content here" in merged


class TestParseAgentsMdFromContent:
    """Tests for parsing AGENTS.md from string content."""

    def test_given_content_string_when_parsed_then_extracts_sections(self) -> None:
        # Given
        content = """# 1. Writing Style

## 1.1 Voice

- 1.1.1: Use active voice.

# 2. Technical Conduct

- 2.0.1: Read code first.
"""

        # When
        doc = parse_agents_md_from_content(content)

        # Then
        assert len(doc.major_sections) == 2
        assert doc.major_sections[0].name == "Writing Style"
        assert doc.major_sections[1].name == "Technical Conduct"

    def test_given_merged_content_when_parsed_then_includes_project_sections(self) -> None:
        # Given
        bundled = """# 1. Bundled Section

- 1.0.1: Bundled rule.
"""
        project = """# 2. Project Section

- 2.0.1: Project rule.
"""
        merged = merge_agents_content(bundled, project)

        # When
        doc = parse_agents_md_from_content(merged)

        # Then
        # Both bundled and project sections should be present
        section_names = [s.name for s in doc.major_sections]
        assert "Bundled Section" in section_names
        assert "Project Section" in section_names


class TestLoadMergedAgentsMd:
    """Tests for load_merged_agents_md function."""

    def test_given_project_agents_md_when_loaded_then_returns_project_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        project_agents = tmp_path / "AGENTS.md"
        project_agents.write_text("# 1. Project Rules\n- 1.0.1: Custom rule.")
        monkeypatch.chdir(tmp_path)

        # When
        path, _content = load_merged_agents_md(tmp_path)

        # Then
        assert path == project_agents

    def test_given_project_agents_md_when_loaded_then_content_contains_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        project_agents = tmp_path / "AGENTS.md"
        project_agents.write_text("# Custom Project Rule")
        monkeypatch.chdir(tmp_path)

        # When
        _path, content = load_merged_agents_md(tmp_path)

        # Then
        # Should contain project content
        assert "# Custom Project Rule" in content
        # Should contain bundled content (Writing Style is in bundled AGENTS.md)
        assert "Writing Style" in content

    def test_given_no_project_agents_md_when_loaded_then_returns_bundled_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given - tmp_path has no AGENTS.md
        monkeypatch.chdir(tmp_path)

        # When
        path, content = load_merged_agents_md(tmp_path)

        # Then
        # Path should be the bundled path (contains "delta" in path)
        assert "delta" in str(path)
        # Content should be bundled only (no project-specific separator)
        assert "Project-Specific Guidelines" not in content
