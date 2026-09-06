import os
import arabic_reshaper
from bidi.algorithm import get_display

from langsmith import traceable

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import ParagraphStyle


# ============================================================
# Design System - الألوان، الخطوط والأنماط
#
# FastAPI wiring note: font paths now default from .env
# (ARABIC_FONT_PATH / ARABIC_BOLD_FONT_PATH) instead of being hardcoded,
# so this still works if the project ever runs on a machine without
# Tahoma at that exact path. Everything else - colors, sizes, styles -
# is untouched, and is exactly the "design tools" (fonts + colors) the
# Report Generator Agent below has access to.
# ============================================================
class DesignSystem:
    def __init__(self,
                 arabic_font_path: str = None,
                 arabic_bold_font_path: str = None):

        arabic_font_path = arabic_font_path or os.getenv(
            "ARABIC_FONT_PATH", "C:/Windows/Fonts/tahoma.ttf"
        )
        arabic_bold_font_path = arabic_bold_font_path or os.getenv(
            "ARABIC_BOLD_FONT_PATH", "C:/Windows/Fonts/tahomabd.ttf"
        )

        self.font_regular_name = "ArabicRegular"
        self.font_bold_name = "ArabicBold"

        if not os.path.exists(arabic_font_path):
            raise FileNotFoundError(
                f"ملف الخط غير موجود: {arabic_font_path}\n"
                "تأكد أن Tahoma موجود بـ C:/Windows/Fonts، أو حدد مسار خط بديل "
                "عبر ARABIC_FONT_PATH في ملف .env"
            )

        pdfmetrics.registerFont(TTFont(self.font_regular_name, arabic_font_path))

        bold_path = arabic_bold_font_path if (arabic_bold_font_path and os.path.exists(arabic_bold_font_path)) else arabic_font_path
        pdfmetrics.registerFont(TTFont(self.font_bold_name, bold_path))

        # الألوان
        self.color_primary = colors.HexColor("#1E3A8A")
        self.color_secondary = colors.HexColor("#3B82F6")
        self.color_accent = colors.HexColor("#F59E0B")
        self.color_text = colors.HexColor("#1F2937")
        self.color_muted = colors.HexColor("#6B7280")
        self.color_success = colors.HexColor("#059669")
        self.color_table_header_bg = colors.HexColor("#1E3A8A")
        self.color_table_header_text = colors.white
        self.color_table_row_alt = colors.HexColor("#F3F4F6")
        self.color_border = colors.HexColor("#D1D5DB")

        # أحجام الخطوط
        self.size_title = 22
        self.size_heading = 15
        self.size_subheading = 12
        self.size_body = 10.5
        self.size_small = 8.5

    def title_style(self):
        return ParagraphStyle(
            "Title",
            fontName=self.font_bold_name,
            fontSize=self.size_title,
            leading=28,
            textColor=self.color_primary,
            alignment=TA_RIGHT,
            spaceAfter=14,
        )

    def heading_style(self):
        return ParagraphStyle(
            "Heading",
            fontName=self.font_bold_name,
            fontSize=self.size_heading,
            leading=20,
            textColor=self.color_secondary,
            alignment=TA_RIGHT,
            spaceBefore=14,
            spaceAfter=8,
        )

    def body_style(self):
        return ParagraphStyle(
            "Body",
            fontName=self.font_regular_name,
            fontSize=self.size_body,
            leading=16,
            textColor=self.color_text,
            alignment=TA_RIGHT,
            spaceAfter=8,
        )

    def muted_style(self):
        return ParagraphStyle(
            "Muted",
            fontName=self.font_regular_name,
            fontSize=self.size_small,
            leading=12,
            textColor=self.color_muted,
            alignment=TA_RIGHT,
        )

    def source_style(self):
        return ParagraphStyle(
            "Source",
            fontName=self.font_regular_name,
            fontSize=self.size_small,
            leading=12,
            textColor=self.color_secondary,
            alignment=TA_RIGHT,
        )


def ar(text: str) -> str:
    """دالة معالجة وتشكيل النص العربي للطباعة في ReportLab"""
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)


# ============================================================
# Report Generator Agent
# ============================================================
class ReportGeneratorAgent:
    def __init__(self, design: DesignSystem = None,
                 arabic_font_path: str = None,
                 arabic_bold_font_path: str = None):
        self.design = design or DesignSystem(
            arabic_font_path=arabic_font_path,
            arabic_bold_font_path=arabic_bold_font_path,
        )

    def _build_header(self, elements: list, title: str, question: str):
        d = self.design
        elements.append(Paragraph(ar(title), d.title_style()))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(ar(f"السؤال: {question}"), d.body_style()))
        elements.append(Spacer(1, 10))
        elements.append(Table(
            [[""]], colWidths=[170 * mm], rowHeights=[1],
            style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, d.color_border)])
        ))
        elements.append(Spacer(1, 10))

    def _build_analysis_section(self, elements: list, analysis_text: str):
        d = self.design
        elements.append(Paragraph(ar("التحليل"), d.heading_style()))
        for line in analysis_text.split("\n"):
            if line.strip():
                elements.append(Paragraph(ar(line.strip()), d.body_style()))

    def _build_calculations_table(self, elements: list, calculations: list):
        if not calculations:
            return
        d = self.design
        elements.append(Paragraph(ar("الحسابات"), d.heading_style()))

        data = [[ar("النتيجة"), ar("التعبير")]]
        for calc in calculations:
            data.append([str(calc.get("result", "")), ar(str(calc.get("expression", "")))])

        table = Table(data, colWidths=[70 * mm, 100 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), d.color_table_header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), d.color_table_header_text),
            ("FONTNAME", (0, 0), (-1, 0), d.font_bold_name),
            ("FONTNAME", (0, 1), (-1, -1), d.font_regular_name),
            ("FONTSIZE", (0, 0), (-1, -1), d.size_body),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, d.color_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, d.color_table_row_alt]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 10))

    def _build_generic_table(self, elements: list, headers: list, rows: list, title: str = "جدول البيانات"):
        d = self.design
        elements.append(Paragraph(ar(title), d.heading_style()))

        data = [[ar(h) for h in headers]] + [[ar(str(cell)) for cell in row] for row in rows]
        col_width = 170 * mm / max(len(headers), 1)
        table = Table(data, colWidths=[col_width] * len(headers))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), d.color_table_header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), d.color_table_header_text),
            ("FONTNAME", (0, 0), (-1, 0), d.font_bold_name),
            ("FONTNAME", (0, 1), (-1, -1), d.font_regular_name),
            ("FONTSIZE", (0, 0), (-1, -1), d.size_small),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, d.color_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, d.color_table_row_alt]),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 10))

    def _build_sources_section(self, elements: list, evidence_chunks: list):
        d = self.design
        elements.append(Paragraph(ar("المصادر"), d.heading_style()))

        seen = {}
        for chunk in evidence_chunks:
            src = chunk.get("source", "غير معروف")
            page = chunk.get("page")
            seen.setdefault(src, set())
            if page:
                seen[src].add(page)

        for src, pages in seen.items():
            pages_str = ", ".join(str(p) for p in sorted(pages)) if pages else "-"
            elements.append(Paragraph(ar(f"• {src} — صفحة: {pages_str}"), d.source_style()))

    def _add_page_footer(self, canvas_obj, doc):
        d = self.design
        canvas_obj.saveState()
        canvas_obj.setFont(d.font_regular_name, 8)
        canvas_obj.setFillColor(d.color_muted)
        canvas_obj.drawCentredString(
            A4[0] / 2, 12 * mm,
            ar(f"صفحة {doc.page}")
        )
        canvas_obj.restoreState()

    # ============================================================
    # دالة التوليد الرئيسية المربوطة بـ LangSmith
    # ============================================================
    @traceable(name="Report Generator: Generate PDF")
    def generate_pdf(self, report_data: dict, output_path: str = "./report.pdf"):
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=20 * mm, leftMargin=20 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
        )

        elements = []
        self._build_header(
            elements,
            title=report_data.get("title", "تقرير النتائج"),
            question=report_data.get("question", ""),
        )

        if report_data.get("final_answer"):
            elements.append(Paragraph(ar("الإجابة النهائية"), self.design.heading_style()))
            for line in report_data["final_answer"].split("\n"):
                if line.strip():
                    elements.append(Paragraph(ar(line.strip()), self.design.body_style()))
            elements.append(Spacer(1, 8))

        if report_data.get("analysis"):
            self._build_analysis_section(elements, report_data["analysis"])

        if report_data.get("calculations"):
            self._build_calculations_table(elements, report_data["calculations"])

        if report_data.get("table"):
            t = report_data["table"]
            self._build_generic_table(elements, t.get("headers", []), t.get("rows", []))

        if report_data.get("evidence"):
            self._build_sources_section(elements, report_data["evidence"])

        doc.build(
            elements,
            onFirstPage=self._add_page_footer,
            onLaterPages=self._add_page_footer,
        )
        return output_path


# ------------------ التشغيل والتجربة ------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = ReportGeneratorAgent()

    sample_data = {
        "title": "تقرير تحليل النتائج",
        "question": "ما هو الشرط الأساسي لإرجاع النص المضغوط؟",
        "final_answer": "يجب أن يكون النص المضغوط أصغر من النص الأصلي، وإلا يتم إرجاع النص الأصلي.",
        "analysis": "دالة ضغط النصوص تعمل عن طريق استبدال تكرار الحروف بعدد مرات تكرارها.\nمثال: aabcccccaaa تصبح a2b1c5a3.",
        "calculations": [{"expression": "len('a2b1c5a3') < len('aabcccccaaa')", "result": True}],
        "evidence": [
            {"source": "compression_notes.pdf", "page": 3, "text": "..."},
            {"source": "compression_notes.pdf", "page": 4, "text": "..."},
        ],
    }

    path = agent.generate_pdf(sample_data, output_path="./sample_report.pdf")
    print(f"✅ تم إنشاء التقرير: {path}")
