"""
Utilities for MIME type detection and content type classification.

This module provides functions to:
- Guess MIME types from file extensions
- Classify content as text, binary, or image based on MIME type
- Handle special cases for text-based formats that don't use 'text/' prefix
"""

import mimetypes

# Initialize mimetypes database
mimetypes.init()

# Extend with additional types that might be missing
mimetypes.add_type("text/x-python", ".py")
mimetypes.add_type("image/webp", ".webp")

# Known text-based MIME types not starting with "text/"
TEXT_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "application/ld+json",
    "application/xhtml+xml",
    "application/x-httpd-php",
    "application/x-sh",
    "application/ecmascript",
    "application/graphql",
    "application/x-www-form-urlencoded",
    "application/yaml",
    "application/toml",
    "application/x-python-code",
    "application/vnd.api+json",
}

# Common text-based MIME type patterns
TEXT_MIME_PATTERNS = ("+xml", "+json", "+yaml", "+text")


def guess_mime_type(file_path: str) -> str:
    """
    Guess the MIME type of a file based on its extension.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def is_text_mime_type(mime_type: str) -> bool:
    """Determine if a MIME type represents text content."""
    if not mime_type:
        return False

    # Standard text types
    if mime_type.startswith("text/"):
        return True

    # Known text types
    if mime_type in TEXT_MIME_TYPES:
        return True

    # Common text patterns
    if any(mime_type.endswith(pattern) for pattern in TEXT_MIME_PATTERNS):
        return True

    return False


def is_binary_content(mime_type: str) -> bool:
    """Check if content should be treated as binary."""
    return not is_text_mime_type(mime_type)


def is_image_mime_type(mime_type: str) -> bool:
    """Check if a MIME type represents an image."""
    return mime_type.startswith("image/") and mime_type != "image/svg+xml"


def image_url_to_mime_and_base64(image_url: str) -> tuple[str, str]:
    """
    Extract mime type and base64 data from ImageUrl
    """
    import re

    match = re.match(r"data:(image/[\w.+-]+);base64,(.*)", image_url)
    if not match:
        raise ValueError(f"Invalid image data URI: {image_url[:30]}...")
    mime_type, base64_data = match.groups()
    return mime_type, base64_data
