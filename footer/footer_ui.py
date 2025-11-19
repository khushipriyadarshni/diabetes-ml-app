"""Footer UI injection for Streamlit app."""

import streamlit as st
import pathlib


def inject_footer() -> None:
    """Inject a fixed glass-effect footer into the Streamlit app."""
    css_path = pathlib.Path(__file__).parent / "style.css"
    try:
        with open(css_path, "r", encoding="utf-8") as file:
            css = file.read()
    except FileNotFoundError:
        st.warning("Footer CSS file not found.")
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="custom-footer">
            <div class="footer-row">
                <span class="footer-pill">💡 Diabetes Risk Prediction System</span>
                <span class="footer-pill">© 2025 · All Rights Reserved</span>
                <span class="footer-pill">Educational & Research Use Only</span>
            </div>
            <div class="footer-row">
                <a class="about-link" href="https://example.com/about" target="_blank" rel="noopener">
                    About Us · Learn more about our mission
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

