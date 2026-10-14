"""Optional post-save hook for future training-time consecutive plots.

The current trainer is unchanged. integration/consecutive_training.patch shows
the small opt-in wiring. This hook reads already-saved artifacts, never a model.
"""

from pathlib import Path


def validate_sequence_plot_settings(settings):
    """Validate optional settings without changing evaluation/training state."""
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("sequence_plots must be null or a plotting settings object")
    from dataclasses import fields
    from .sequence_visualization import PlotSettings

    unknown = set(settings) - {field.name for field in fields(PlotSettings)}
    if unknown:
        raise ValueError(f"Unknown sequence plot settings: {sorted(unknown)}")
    # Dataset-dependent validation runs inside render_sequence before any output.
    PlotSettings(**settings)
    return dict(settings)


def render_saved_split(output_dir, settings=None):
    """Return output file paths after rendering an already-saved scored split.

    settings=None disables the optional hook; {} enables defaults. Models with
    no saved edge scores have no graph to render and return an empty list.
    """
    settings = validate_sequence_plot_settings(settings)
    if settings is None:
        return []
    destination = Path(output_dir)
    score_path = destination / "edge_scores.npz"
    if not score_path.is_file():
        return []
    from .sequence_visualization import load_sequence, render_sequence

    manifest = render_sequence(load_sequence(score_path),
                               destination / "figures" / "consecutive", settings)
    return [*manifest["figures"], manifest["manifest_path"], manifest["window_index_csv"]]
