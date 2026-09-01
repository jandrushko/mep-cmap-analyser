"""
mep_cmap.docx_writer
~~~~~~~~~~~~~~~~~~~~
Rendered blocks in, Word document bytes out, using nothing but the standard
library.

Why not python-docx
-------------------
It depends on lxml, which is a compiled wheel. Adding it would put a binary
dependency into every install of a tool whose runtime profile is otherwise
pure Python, and into three PyInstaller specs that would each have to be
taught about it. A .docx is a zip of XML, and the part of the format a methods
draft needs -- headings, paragraphs, formatted runs -- is small and stable.

The cost of the alternative is already visible in this codebase: openpyxl is a
dev-only dependency for the same class of reason, and the consequence is that
the filled TMS-RAT workbook is never written for anyone who installed the tool
normally. A document output that only exists for developers is not an output.

Scope
-----
This module knows about blocks, not about TMS-RAT. It renders whatever
``mep_cmap.tms_rat.methods_blocks`` produces, in the same shape any other
report could produce.

It returns BYTES rather than writing a file, so a document can be built and
asserted on without a temp directory, and so the caller owns the I/O.
"""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

#: A fixed timestamp on every zip entry, so the same blocks always produce the
#: same bytes. A document that differs run to run cannot be diffed, and a
#: reproducibility claim is easier to make than to check.
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)

#: How a gap is marked: bold, dark red, highlighted. The marker text itself is
#: left exactly as it is, so the draft is still searchable for "[TMS-RAT" by
#: someone who has turned formatting off or pasted into a plain editor. The
#: formatting is there for the reader who is skimming, not instead of the text.
_GAP_RPR = ('<w:rPr><w:b/><w:color w:val="C00000"/>'
            '<w:highlight w:val="yellow"/></w:rPr>')

_HEADING_STYLE = {1: "Heading1", 2: "Heading2", 3: "Heading3"}

_W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

#: A4 with 2 cm margins, in twentieths of a point.
_SECTPR = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
           'w:left="1134" w:header="709" w:footer="709" w:gutter="0"/>'
           '</w:sectPr>')

_DOCUMENT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<w:document ' + _W_NS + '><w:body>@BODY@</w:body></w:document>')

def _style(style_id, name, *, ppr="", rpr="", default=False, priority=None):
    """One paragraph style definition.

    The styles are REAL Word heading styles, not direct formatting that merely
    looks like one. A manuscript is going to be pasted into a journal template,
    and a template restyles Heading 1; it cannot restyle 16-point bold text
    that happens to sit on its own line. The navigation pane and any generated
    table of contents work for the same reason.
    """
    bits = [f'<w:name w:val="{name}"/>']
    if not default:
        bits.append('<w:basedOn w:val="Normal"/><w:next w:val="Normal"/>')
    if priority is not None:
        bits.append(f'<w:uiPriority w:val="{priority}"/>')
    bits.append("<w:qFormat/>")
    if ppr:
        bits.append(f"<w:pPr>{ppr}</w:pPr>")
    if rpr:
        bits.append(f"<w:rPr>{rpr}</w:rPr>")
    dflt = ' w:default="1"' if default else ""
    return (f'<w:style w:type="paragraph"{dflt} w:styleId="{style_id}">'
            + "".join(bits) + "</w:style>")


_STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<w:styles ' + _W_NS + '>'
           '<w:docDefaults><w:rPrDefault><w:rPr>'
           '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
           '<w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-GB"/>'
           '</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>'
           '<w:spacing w:after="160" w:line="276" w:lineRule="auto"/>'
           '</w:pPr></w:pPrDefault></w:docDefaults>'
           + _style("Normal", "Normal", default=True)
           + _style("Heading1", "heading 1", priority=9,
                    ppr=('<w:keepNext/><w:spacing w:before="240" w:after="120"/>'
                         '<w:outlineLvl w:val="0"/>'),
                    rpr='<w:b/><w:sz w:val="32"/><w:szCs w:val="32"/>')
           + _style("Heading2", "heading 2", priority=9,
                    ppr=('<w:keepNext/><w:spacing w:before="240" w:after="120"/>'
                         '<w:outlineLvl w:val="1"/>'),
                    rpr='<w:b/><w:sz w:val="26"/><w:szCs w:val="26"/>')
           + _style("Heading3", "heading 3", priority=9,
                    ppr=('<w:keepNext/><w:spacing w:before="200" w:after="80"/>'
                         '<w:outlineLvl w:val="2"/>'),
                    rpr='<w:b/><w:i/><w:sz w:val="24"/><w:szCs w:val="24"/>')
           + _style("ListParagraph", "List Paragraph", priority=34,
                    ppr=('<w:spacing w:after="60"/>'
                         '<w:ind w:left="360" w:hanging="360"/>'))
           + "</w:styles>")


def _clean(text):
    """Text an XML parser will accept.

    A control character is not representable in XML 1.0 at all, and one
    arriving from a filename or an instrument string would produce a file Word
    offers to repair rather than open. Tabs and newlines survive as spaces
    because a paragraph is a paragraph.
    """
    out = []
    for ch in str(text or ""):
        if ch in "\t\r\n":
            out.append(" ")
        elif ord(ch) >= 0x20:
            out.append(ch)
    return "".join(out)


def _run_xml(run):
    """One run. xml:space is preserved because runs abut.

    "Stimulator: " and its value are separate runs so the value can be
    highlighted, and without xml:space the trailing space is dropped and the
    two words run together.
    """
    props = _GAP_RPR if run.get("gap") else ""
    text = escape(_clean(run.get("text")))
    return f'<w:r>{props}<w:t xml:space="preserve">{text}</w:t></w:r>'


def _paragraph_xml(block):
    kind = block.get("kind", "para")
    runs = list(block.get("runs") or [])
    if kind == "heading":
        style = _HEADING_STYLE.get(block.get("level") or 2, "Heading2")
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    elif kind == "bullet":
        # A literal bullet character in a hanging indent, rather than a real
        # Word list. A list would need a numbering part and its own
        # relationship, which is a lot of format surface for a section the
        # author deletes before submitting.
        ppr = '<w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr>'
        runs = [{"text": "\u2022  ", "gap": False}] + runs
    else:
        ppr = ""
    return "<w:p>" + ppr + "".join(_run_xml(r) for r in runs) + "</w:p>"


def document_bytes(blocks):
    """The blocks as a .docx, in memory.

    Every part is written explicitly rather than copied from a template file,
    so there is no asset to ship, nothing to go missing from a wheel, and the
    whole document is readable in this module.
    """
    body = "".join(_paragraph_xml(b) for b in (blocks or [])) + _SECTPR
    parts = (("[Content_Types].xml", _CONTENT_TYPES),
             ("_rels/.rels", _RELS),
             ("word/_rels/document.xml.rels", _DOC_RELS),
             ("word/styles.xml", _STYLES),
             ("word/document.xml", _DOCUMENT.replace("@BODY@", body)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)
    return buf.getvalue()
