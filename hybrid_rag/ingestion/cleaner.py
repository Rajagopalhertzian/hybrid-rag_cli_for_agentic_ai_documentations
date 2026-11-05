"""
MDX/JSX cleaner for markdown files.

Strips:
- JSX components (<Component />, <Component>...</Component>)
- JSX expressions ({...})
- Import/export statements
- Frontmatter (--- YAML ---)

Keeps:
- Headings (# ## ###)
- Code blocks (```)
- Tables
- Lists
- Plain text
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CleanResult:
    """Result of cleaning a markdown file."""

    content: str
    heading_path: tuple[str, ...]
    frontmatter: dict


# Patterns for cleaning
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
JSX_COMPONENT_PATTERN = re.compile(r"<[A-Z][A-Za-z0-9]*\b[^>]*/>")
JSX_OPEN_TAG_PATTERN = re.compile(r"<[A-Z][A-Za-z0-9]*\b[^>]*>")
JSX_CLOSE_TAG_PATTERN = re.compile(r"</[A-Z][A-Za-z0-9]*>")
JSX_EXPRESSION_PATTERN = re.compile(r"\{[^}]+\}")
IMPORT_PATTERN = re.compile(r"^import\s+.*$", re.MULTILINE)
EXPORT_PATTERN = re.compile(r"^export\s+.*$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def extract_frontmatter(content: str) -> tuple[str, dict]:
    """Extract and parse YAML frontmatter."""
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return content, {}

    frontmatter_text = match.group(1)
    content_without_frontmatter = content[match.end() :]

    # Simple YAML parsing for common keys
    frontmatter = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip("\"'")

    return content_without_frontmatter, frontmatter


def strip_jsx_components(content: str) -> str:
    """Remove JSX components (self-closing and paired tags)."""
    # Self-closing: <Component />
    content = JSX_COMPONENT_PATTERN.sub("", content)

    # Paired tags: <Component>...</Component>
    # This is trickier - we need to handle nested components
    # Simple approach: remove opening and closing tags
    content = JSX_OPEN_TAG_PATTERN.sub("", content)
    content = JSX_CLOSE_TAG_PATTERN.sub("", content)

    return content


def strip_jsx_expressions(content: str) -> str:
    """Remove JSX expressions {...}."""
    return JSX_EXPRESSION_PATTERN.sub("", content)


def strip_imports_exports(content: str) -> str:
    """Remove import/export statements."""
    content = IMPORT_PATTERN.sub("", content)
    content = EXPORT_PATTERN.sub("", content)
    return content


def extract_heading_path(content: str) -> tuple[str, ...]:
    """Extract heading hierarchy from markdown."""
    headings = HEADING_PATTERN.findall(content)
    return tuple(h[1].strip() for h in headings)


def clean_markdown(content: str) -> CleanResult:
    """
    Clean markdown content by stripping MDX/JSX artifacts.

    Returns cleaned content and heading path.
    """
    # Extract frontmatter first
    content, frontmatter = extract_frontmatter(content)

    # Extract heading path before cleaning (from original)
    heading_path = extract_heading_path(content)

    # Strip JSX components
    content = strip_jsx_components(content)

    # Strip JSX expressions
    content = strip_jsx_expressions(content)

    # Strip imports/exports
    content = strip_imports_exports(content)

    # Clean up extra blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return CleanResult(
        content=content,
        heading_path=heading_path,
        frontmatter=frontmatter,
    )