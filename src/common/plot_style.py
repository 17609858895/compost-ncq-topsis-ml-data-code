import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


def _register_arial_fallback():
    """Prefer Arial (present on the author's machine); fall back to the
    metric-compatible Liberation Sans so renders on other systems match."""
    from matplotlib import font_manager
    for base in ("/usr/share/fonts/truetype/liberation2",
                 "/usr/share/fonts/truetype/liberation"):
        for fn in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
                   "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
            p = os.path.join(base, fn)
            if os.path.exists(p):
                try:
                    font_manager.fontManager.addfont(p)
                except Exception:
                    pass


def configure_publication_style():
    _register_arial_fallback()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "savefig.dpi": 600,
        "figure.dpi": 160,
        "axes.linewidth": 1.55,
        "axes.labelweight": "bold",
        "axes.labelsize": 15.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 11.5,
        "xtick.major.size": 7.2,
        "ytick.major.size": 7.2,
        "xtick.major.width": 1.55,
        "ytick.major.width": 1.55,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, *, keep_box=False):
    if getattr(ax, "name", "") == "polar":
        ax.tick_params(labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        return
    ax.grid(False)
    if not keep_box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_linewidth(1.55)
    ax.spines["bottom"].set_linewidth(1.55)
    ax.tick_params(axis="x", which="major", direction="out", length=7.2,
                   width=1.55, bottom=True, top=False, labelsize=12.5)
    ax.tick_params(axis="y", which="major", direction="out", length=7.2,
                   width=1.55, left=True, right=False, labelsize=12.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _scale_text(text, factor):
    size = text.get_fontsize()
    if size:
        text.set_fontsize(size * factor)


def scale_figure_text(fig, factor=1.28):
    for ax in fig.axes:
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + list(ax.texts)
        ):
            _scale_text(item, factor)
            item.set_fontweight(item.get_fontweight() or "bold")
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                _scale_text(item, factor)
            if legend.get_title() is not None:
                _scale_text(legend.get_title(), factor)
    for legend in fig.legends:
        for item in legend.get_texts():
            _scale_text(item, factor)
        if legend.get_title() is not None:
            _scale_text(legend.get_title(), factor)


def _ensure_min_text_size(text, min_size):
    size = text.get_fontsize()
    if size and size < min_size:
        text.set_fontsize(min_size)
    text.set_fontweight(text.get_fontweight() or "bold")


def enforce_readable_labels(fig):
    """Keep reader-facing labels legible without enlarging dense cell annotations."""
    for ax in fig.axes:
        is_colorbar = hasattr(ax, "_colorbar")
        if is_colorbar:
            for item in [ax.xaxis.label, ax.yaxis.label]:
                _ensure_min_text_size(item, 10.8)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                _ensure_min_text_size(item, 9.8)
            continue
        for item in [ax.xaxis.label, ax.yaxis.label]:
            _ensure_min_text_size(item, 13.2)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            _ensure_min_text_size(item, 11.2)
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                _ensure_min_text_size(item, 10.2)
            if legend.get_title() is not None:
                _ensure_min_text_size(legend.get_title(), 10.5)
    for legend in fig.legends:
        for item in legend.get_texts():
            _ensure_min_text_size(item, 10.2)
        if legend.get_title() is not None:
            _ensure_min_text_size(legend.get_title(), 10.5)


def clear_infigure_titles(fig):
    for ax in fig.axes:
        ax.set_title("", loc="left")
        ax.set_title("", loc="center")
        ax.set_title("", loc="right")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")


def polish_figure(fig, *, boxed_axes=(), text_scale=1.28):
    boxed = set(boxed_axes)
    for ax in fig.axes:
        if not ax.axison:
            continue
        if hasattr(ax, "_colorbar"):
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            continue
        style_axis(ax, keep_box=ax in boxed)
    clear_infigure_titles(fig)
    scale_figure_text(fig, factor=text_scale)
    enforce_readable_labels(fig)


def _bbox_for_axes(fig, axes, renderer=None):
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax is not None]
    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox.expanded(1.06, 1.08)


def save_figure_with_subfigures(fig, out_dir, stem, panel_axes):
    os.makedirs(out_dir, exist_ok=True)
    sub_dir = os.path.join(out_dir, "subfigures")
    os.makedirs(sub_dir, exist_ok=True)
    def safe_save(path, **kwargs):
        try:
            fig.savefig(path, **kwargs)
        except PermissionError:
            root, ext = os.path.splitext(path)
            alt = f"{root}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
            fig.savefig(alt, **kwargs)
            print(f"Saved alternate file because the original is locked: {alt}")

    safe_save(os.path.join(out_dir, f"{stem}.png"), bbox_inches="tight", dpi=600)
    safe_save(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    # Draw once and reuse the renderer for every panel-group bbox below --
    # repeated fig.canvas.draw() calls are by far the most expensive part of
    # this function for dense panels (e.g. SHAP beeswarms), and nothing in
    # the figure changes between panel exports.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = "abcdefghijklmnopqrstuvwxyz"
    for idx, axes in enumerate(panel_axes):
        bbox = _bbox_for_axes(fig, axes, renderer=renderer)
        label = labels[idx]
        safe_save(os.path.join(sub_dir, f"{stem}_{label}.png"), bbox_inches=bbox, dpi=600)
        safe_save(os.path.join(sub_dir, f"{stem}_{label}.pdf"), bbox_inches=bbox)
    with open(os.path.join(sub_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {stem} subfigures\n\n")
        f.write("Subfigures were exported from the same final combined figure.\n")


def _safe_save_figure(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        root, ext = os.path.splitext(path)
        alt = f"{root}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
        fig.savefig(alt, **kwargs)
        print(f"Saved alternate file because the original is locked: {alt}")


def save_combined_figure(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    _safe_save_figure(fig, os.path.join(out_dir, f"{stem}.png"), bbox_inches="tight", dpi=600)
    _safe_save_figure(fig, os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")


def save_independent_subfigures(out_dir, stem, panel_builders):
    """Render each subfigure on its own canvas instead of cropping the combined figure.

    Each builder is a tuple: (label, figsize, draw_func, boxed_axes). draw_func
    receives the new Figure and must return the axes that need box styling.
    """
    sub_dir = os.path.join(out_dir, "subfigures")
    os.makedirs(sub_dir, exist_ok=True)
    for label, figsize, draw_func, boxed_axes in panel_builders:
        panel_fig = plt.figure(figsize=figsize, dpi=300)
        result = draw_func(panel_fig)
        if isinstance(result, dict):
            axes = result.get("axes", ())
            boxed_axes = result.get("boxed_axes", boxed_axes)
        else:
            axes = result
        if boxed_axes is None:
            boxed_axes = ()
        elif boxed_axes == "returned":
            boxed_axes = axes if isinstance(axes, (list, tuple)) else (axes,)
        polish_figure(panel_fig, boxed_axes=boxed_axes, text_scale=1.06)
        _safe_save_figure(panel_fig, os.path.join(sub_dir, f"{stem}_{label}.png"),
                          bbox_inches="tight", dpi=600)
        _safe_save_figure(panel_fig, os.path.join(sub_dir, f"{stem}_{label}.pdf"),
                          bbox_inches="tight")
        plt.close(panel_fig)
    with open(os.path.join(sub_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {stem} subfigures\n\n")
        f.write("Subfigures were independently rendered on standalone canvases, not cropped from the combined figure.\n")
