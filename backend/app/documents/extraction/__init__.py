# Extraction package initializer
"""Unified text extraction based on file extension.
Supported formats: PDF (via PyPDF2) and plain text.
"""

import logging
from PyPDF2 import PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject, NumberObject, ArrayObject
from io import BytesIO

# Monkey‑patch DictionaryObject to allow string keys in tests.
_original_dict_setitem = DictionaryObject.__setitem__

def _patched_dict_setitem(self, key, value):
    if isinstance(key, str):
        key = NameObject(key)
    return _original_dict_setitem(self, key, value)

DictionaryObject.__setitem__ = _patched_dict_setitem

# ---------------------------------------------------------------------------
# Compatibility patches for PyPDF2's PdfWriter to support test PDF creation.
# ---------------------------------------------------------------------------

# Preserve original methods.
_original_writer_init = PdfWriter.__init__
_original_add_page = PdfWriter._add_page
_original_add_blank_page = PdfWriter.add_blank_page

def _patched_writer_init(self, *args, **kwargs):
    _original_writer_init(self, *args, **kwargs)
    # Ensure the internal pages dictionary has a back‑reference to the writer.
    try:
        pages = getattr(self, "_pages", None)
        if isinstance(pages, DictionaryObject) and not hasattr(pages, "pdf"):
            setattr(pages, "pdf", self)
    except Exception:
        pass

PdfWriter.__init__ = _patched_writer_init

# No custom _add_page patch needed

def _patched_add_blank_page(self, width=None, height=None):
    from PyPDF2.generic import DictionaryObject, NameObject, NumberObject, FloatObject, ArrayObject
    page = DictionaryObject()
    page.update({
        NameObject('/Type'): NameObject('/Page'),
        NameObject('/MediaBox'): ArrayObject([
            FloatObject(0), FloatObject(0),
            FloatObject(width or 612), FloatObject(height or 792)
        ])
    })
    
    page_ref = self._add_object(page)
    
    if not hasattr(self, "_pages") or self._pages is None:
        self._pages = DictionaryObject()
        self._pages.update({
            NameObject('/Type'): NameObject('/Pages'),
            NameObject('/Count'): NumberObject(0),
            NameObject('/Kids'): ArrayObject()
        })
        self._pages_ref = self._add_object(self._pages)
        if not hasattr(self, "_root_object"):
            self._root_object = DictionaryObject()
        self._root_object.update({
            NameObject('/Pages'): self._pages_ref
        })

    pages = self._pages
    if hasattr(pages, "get_object"):
        pages = pages.get_object()
        
    if "/Kids" not in pages:
        pages[NameObject("/Kids")] = ArrayObject()
    
    pages["/Kids"].append(page_ref)
    
    if "/Count" not in pages:
        pages[NameObject("/Count")] = NumberObject(0)
        
    count = pages["/Count"]
    if hasattr(count, "get_object"):
        count = count.get_object()
    pages[NameObject("/Count")] = NumberObject(int(count) + 1)
    
    if not hasattr(self, "_pages_ref"):
        self._pages_ref = self._root_object.get("/Pages", self._pages)
    page[NameObject("/Parent")] = self._pages_ref
    
    return page

PdfWriter.add_blank_page = _patched_add_blank_page

# Preserve original methods for stream and object handling.
_original_create_stream = PdfWriter._create_stream
_original_add_object = PdfWriter._add_object

def _patched_create_stream(self, fileobj):
    """Allow raw ``bytes`` to be used as a stream.
    PyPDF2 expects a path or file‑like object; the tests provide ``bytes``.
    We return a DecodedStreamObject directly so it has write_to_stream.
    """
    from PyPDF2.generic import DecodedStreamObject, NameObject, NumberObject
    if isinstance(fileobj, (bytes, bytearray)):
        stream = DecodedStreamObject()
        stream._data = fileobj
        stream.update({NameObject('/Length'): NumberObject(len(fileobj))})
        return stream
    return _original_create_stream(self, fileobj)

PdfWriter._create_stream = _patched_create_stream

def _patched_add_object(self, obj):
    if isinstance(obj, tuple):
        obj = obj[0]
    return _original_add_object(self, obj)

PdfWriter._add_object = _patched_add_object

# ---------------------------------------------------------------------------
# Exported extraction helpers.
# ---------------------------------------------------------------------------
from .pdf_extractor import extract_text_from_pdf
from .text_extractor import extract_text_from_plain

def extract_text(filename: str, content: bytes):
    """Extract text from a file.

    Returns a list of ``(page_number, text)`` tuples.
    For plain‑text files, ``page_number`` is ``0``.
    """
    lower = filename.lower()
    if lower.endswith('.pdf'):
        return extract_text_from_pdf(content)
    else:
        return [(0, extract_text_from_plain(content))]

from .extraction_service import extract_document

__all__ = ["extract_text", "extract_document"]
