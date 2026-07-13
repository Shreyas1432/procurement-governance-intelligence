"""
Shared design tokens + CSS injector for the PGI dashboard.

Source of truth: docs/design/PGI.dc.html (Claude Design export). Tokens below
are extracted directly from that file's inline styles. Do not hand-tune
colors here without checking the design first.

Palette (primitive):
  Ink        #221F1D   primary text
  Ink-muted  #55534F   secondary text
  Ink-faint  #6B6B6B   tertiary text / captions
  Ink-quiet  #9B9791   labels, scope tags, placeholders
  Border     #E4E1D9   card borders
  Border-lt  #EDEAE2 / #F0EEE8   inner dividers
  Surface    #FFFFFF   card background
  Canvas     #F9F8F6 / #FBFAF8   page background
  Panel      #F3F1EC   neutral panel (methodology-note grey)

  Primary (navy)   #3B5998 / tint #EBEFF6   brand, links, RQ1
  Sage (positive)  #7A9B6F / dark #4F6B47 / tint #EEF2EA   methodology, RQ3
  Rust (caution)   #C85A54 / dark #8C3C37 / tint #FBEAE8   risk, caveats, RQ2
  Amber (accent)   #B98A3E   secondary agreement tier (RQ3 ensemble chart)

Typography: Merriweather (serif) for headings, Inter (sans) for body/UI.
"""
from __future__ import annotations

import streamlit as st

# tokens
INK = "#221F1D"
INK_MUTED = "#55534F"
INK_FAINT = "#6B6B6B"
INK_QUIET = "#9B9791"
BORDER = "#E4E1D9"
BORDER_LIGHT = "#EDEAE2"
BORDER_LIGHTER = "#F0EEE8"
SURFACE = "#FFFFFF"
CANVAS = "#F9F8F6"
CANVAS_ALT = "#FBFAF8"
PANEL = "#F3F1EC"

PRIMARY = "#3B5998"
PRIMARY_TINT = "#EBEFF6"
SAGE = "#7A9B6F"
SAGE_DARK = "#4F6B47"
SAGE_TINT = "#EEF2EA"
RUST = "#C85A54"
RUST_DARK = "#8C3C37"
RUST_TINT = "#FBEAE8"
AMBER = "#B98A3E"

FONT_SERIF = "'Merriweather', serif"
FONT_SANS = "'Inter', sans-serif"

RISK_CAVEAT = (
    "Every risk label here describes a **procedural-recording irregularity**, "
    "not corruption. This is not a corruption-detection tool. Regional "
    "breakdowns draw on small samples and should be read with that limit in mind."
)


def inject_base_css() -> None:
    """Global CSS: fonts, background, and Streamlit-chrome overrides. Call
    once per page, after st.set_page_config()."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_SANS};
        }}
        h1, h2, h3 {{
            font-family: {FONT_SERIF} !important;
        }}
        .stApp {{
            background-color: {CANVAS};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        /* Hide Streamlit's own chrome to match the design's quiet shell:
           hamburger menu / Deploy button, the header's default
           background, and the "Made with Streamlit" footer. NOTE: do NOT hide
           div[data-testid="stSidebarNav"], that container is also where
           st.navigation() renders its own page menu (app.py's router), so
           hiding it removes the real navigation, not just Streamlit's legacy
           auto-discovered pages/ list. */
        #MainMenu {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        footer {{ visibility: hidden; }}
        div[data-testid="stDecoration"] {{ display: none; }}
        div[data-testid="stStatusWidget"] {{ visibility: hidden; }}
        .block-container {{ padding-top: 2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def scope_tag(text: str, color: str = PRIMARY) -> str:
    """Small uppercase eyebrow label, e.g. 'RQ1' or 'Admin only'."""
    return (
        f"<div style='font-size:12px;font-weight:700;letter-spacing:0.05em;"
        f"text-transform:uppercase;color:{color};margin-bottom:8px;'>{text}</div>"
    )


def page_title(title: str, size: str = "36px") -> str:
    return (
        f"<h1 style=\"font-family:{FONT_SERIF};font-size:{size};"
        f"color:{INK};margin:0 0 10px;\">{title}</h1>"
    )
