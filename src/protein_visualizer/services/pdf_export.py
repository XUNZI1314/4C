from io import BytesIO


def build_simple_pdf(report_text: str) -> bytes:
    safe_lines = [line.replace('(', '[').replace(')', ']') for line in report_text.splitlines()]
    content_stream = "BT\n/F1 10 Tf\n50 780 Td\n14 TL\n"
    first = True
    for line in safe_lines:
        escaped = line.replace('\\', '\\\\')
        escaped = escaped.replace('(', '\\(').replace(')', '\\)')
        if first:
            content_stream += f"({escaped}) Tj\n"
            first = False
        else:
            content_stream += f"T* ({escaped}) Tj\n"
    content_stream += "ET"
    content_bytes = content_stream.encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")
    objects.append(f"4 0 obj << /Length {len(content_bytes)} >> stream\n".encode("latin-1") + content_bytes + b"\nendstream endobj\n")
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    pdf = BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(pdf.tell())
        pdf.write(obj)

    xref_pos = pdf.tell()
    pdf.write(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.write(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )
    return pdf.getvalue()
