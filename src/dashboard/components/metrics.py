"""Styled KPI metric and page-header components, the shared design system
for the PGI dashboard. One palette, one header treatment, one KPI card style
across every page (Task 7 consolidation)."""
import streamlit as st

# Design tokens: primitive -> semantic. Single source of truth for color across all pages.
# Primary brand color matches .streamlit/config.toml's primaryColor (#0f766e).
COLOR_PRIMARY = "#0f766e"       # teal-700, brand primary
COLOR_PRIMARY_DARK = "#0d9488"  # teal-600, gradient partner
COLOR_RISK_HIGH = "#ef4444"     # red-500
COLOR_RISK_MEDIUM = "#f59e0b"   # amber-500
COLOR_RISK_LOW = "#22c55e"      # green-500
COLOR_INFO = "#3b82f6"          # blue-500
COLOR_NEUTRAL = "#64748b"       # slate-500

PAGE_HEADER_CSS = """
<style>
  .pgi-page-header {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
    color: white; padding: 18px 24px; border-radius: 8px; margin-bottom: 20px;
  }
  .pgi-page-header h2 { margin: 0; font-size: 1.4rem; font-weight: 700; }
  .pgi-page-header p  { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.85rem; }
</style>
"""


def render_page_header(title: str, subtitle: str) -> None:
    """One consistent header treatment for every page: single teal brand
    gradient, same typography scale. Replaces the previous per-page
    rainbow of gradients (teal/blue/purple/red) with one design system."""
    st.markdown(PAGE_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""<div class='pgi-page-header'><h2>{title}</h2><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """Render a row of metric cards.
    items: list of (label, value, delta) tuples
    """
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def kpi_row_styled(items):
    """Render styled KPI cards with colored backgrounds.
    items: list of dicts {label, value, delta, color}
    Colors: 'red'|'amber'|'green'|'blue'|'default'
    """
    COLORS = {
        "red":     ("", "#fef2f2", "#ef4444"),
        "amber":   ("", "#fffbeb", "#f59e0b"),
        "green":   ("", "#f0fdf4", "#22c55e"),
        "blue":    ("", "#eff6ff", "#3b82f6"),
        "teal":    ("", "#f0fdfa", "#0f766e"),
        "default": ("", "#f8fafc", "#64748b"),
    }
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        icon, bg, accent = COLORS.get(item.get("color", "default"), COLORS["default"])
        delta = item.get("delta")
        delta_html = f"<span style='color:{accent};font-size:0.75rem'>{delta}</span>" if delta else ""
        col.markdown(f"""
<div style='background:{bg};border-left:4px solid {accent};border-radius:6px;
            padding:14px 16px;margin-bottom:4px'>
  <div style='font-size:0.75rem;color:#64748b;font-weight:600;text-transform:uppercase;
              letter-spacing:0.05em'>{icon + " " if icon else ""}{item['label']}</div>
  <div style='font-size:1.6rem;font-weight:700;color:{accent}'>{item['value']}</div>
  {delta_html}
</div>
""", unsafe_allow_html=True)
