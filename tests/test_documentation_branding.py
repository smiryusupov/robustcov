"""Regression checks for the public documentation identity and landing page."""

from __future__ import annotations

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BRAND = DOCS / "_static" / "brand"


def _is_png(path: Path) -> bool:
    return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def test_brand_assets_are_present_and_valid_png_files():
    for name in (
        "robustcov-mark.png",
        "robustcov-lockup.png",
        "robustcov-favicon.png",
    ):
        path = BRAND / name
        assert path.is_file(), path
        assert path.stat().st_size > 1_000, path
        assert _is_png(path), path
    assert _png_size(BRAND / "robustcov-mark.png") == (512, 512)
    assert _png_size(BRAND / "robustcov-favicon.png") == (128, 128)


def test_furo_configuration_uses_the_brand_assets_and_palette():
    text = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert "html_logo = '_static/brand/robustcov-mark.png'" in text
    assert "html_favicon = '_static/brand/robustcov-favicon.png'" in text
    assert '"color-brand-primary": "#123f7a"' in text
    assert '"color-brand-content": "#2f6fda"' in text


def test_landing_page_has_one_clear_product_message_and_three_entry_paths():
    text = (DOCS / "index.rst").read_text(encoding="utf-8")
    assert "Robust multivariate geometry for difficult data." in text
    assert "robustcov-hero" in text
    assert 'class="robustcov-hero-copy"' in text
    assert "Start with the quickstart" in text
    assert "Choose a method" in text
    assert "Browse examples" in text
    assert "Choose a path" in text
    assert "What RobustCov provides" in text
    assert "A 60-second start" in text
    assert text.count('class="robustcov-path-card"') == 3
    assert text.count('class="robustcov-capability"') == 4


def test_readme_and_stylesheet_use_the_same_identity():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    css = (DOCS / "_static" / "custom.css").read_text(encoding="utf-8")
    assert "docs/_static/brand/robustcov-lockup.png" in readme
    for selector in (
        ".sidebar-brand",
        ".robustcov-hero",
        ".robustcov-hero-copy",
        ".robustcov-wordmark-robust",
        ".robustcov-wordmark-cov",
        ".robustcov-path-grid",
        ".robustcov-capability-grid",
    ):
        assert selector in css


def test_brand_lockups_are_centered_and_stacked():
    css = (DOCS / "_static" / "custom.css").read_text(encoding="utf-8")
    assert ".sidebar-brand {" in css
    assert "text-align: center;" in css
    hero_block = css.split(".robustcov-hero-brand {", 1)[1].split("}", 1)[0]
    assert "flex-direction: column;" in hero_block
    assert "align-items: center;" in hero_block
    assert "text-align: center;" in hero_block
    assert "margin: 1.45rem auto 0;" in css
    assert "justify-content: center;" in css
