# backend/app/documents/evidence/validator.py

"""Evidence validation logic for retrieved document chunks.

The validator ensures that the retrieved evidence satisfies basic
requirements before it is used downstream. Currently the validation
rules are minimal and can be extended in the future.
"""

from typing import List, Any

from .exceptions import EvidenceValidatorError


class EvidenceValidator:
    """Validate a list of evidence chunks.

    The validator checks that:
    - The evidence list is not empty.
    - Each entry is a dictionary (or an object with ``__dict__``) that
      contains at least a ``'content'`` key/value.
    Further validation rules can be added as needed.
    """

    @staticmethod
    def validate(evidence: List[Any]) -> None:
        """Validate the provided evidence.

        Args:
            evidence: A list of retrieved evidence items.

        Raises:
            EvidenceValidatorError: If the evidence is empty or does not
                contain the required fields.
        """
        if not evidence:
            raise EvidenceValidatorError("No evidence retrieved for the query.")

        for idx, item in enumerate(evidence):
            # Accept both dicts and objects with attributes.
            if isinstance(item, dict):
                if "content" not in item:
                    raise EvidenceValidatorError(
                        f"Evidence item at index {idx} missing 'content' field."
                    )
            else:
                # Fallback to attribute access.
                if not hasattr(item, "content"):
                    raise EvidenceValidatorError(
                        f"Evidence item at index {idx} missing 'content' attribute."
                    )

        # If all checks pass, simply return.
        return None
