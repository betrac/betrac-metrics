"""Matcher protocol and implementations for medical concept extraction."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Matcher(Protocol):
    """Protocol for medical concept matchers.

    Any matcher must implement .match(text) returning a list of concept groups.
    Each group is a list of dicts with at least these keys:
        - 'cui' (str): Concept Unique Identifier (e.g., MeSH descriptor ID "D009203")
        - 'term' (str): The matched text (e.g., "myocardial infarction")
        - 'similarity' (float): Match confidence score (1.0 for exact keyword matches)
        - 'source' (str): Origin of the match (e.g., "mesh", "scispacy")

    This format is compatible with QuickUMLS output.
    """

    def match(self, text: str) -> list[list[dict]]: ...
