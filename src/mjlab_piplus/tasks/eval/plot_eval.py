"""Plot the per-episode CSVs an evaluation run wrote.

    uv run python -m mjlab_piplus.tasks.eval.plot_eval --runs logs/eval/<run>
    uv run python -m mjlab_piplus.tasks.eval.plot_eval --runs A B --labels before after

One PNG per command type, plus an ``overview`` comparing the types against each
other. Up to three runs can be overlaid to compare policies; past three the
scatter panels stop being readable (colors that far apart cannot be told apart
under colour-vision deficiency), so the script refuses rather than cycling hues.

The figures are a *view* of the CSVs, not a replacement: the numbers stay in
``episodes_<type>.csv`` and ``summary.json`` next to them.

What each panel is for
----------------------
The form follows the question each command type asks:

* **velocity** -- commanded against achieved, one panel per axis, with the
  ``y = x`` line the policy is trying to sit on; below it the error
  distribution, which is what the summary's mean collapses.
* **goal_pose** / **path** -- time to goal against how far away the goal was,
  because a slow episode over 8 m is not the same result as a slow one over 1 m.
* **footstep** -- the distribution of per-episode placement error, plus the
  linear error against the angular one, which says whether the run that puts
  the foot in the wrong place is the run that puts it at the wrong angle.
* **kick** -- commanded strength against the speed the ball actually reached,
  and how far off the aim was.
* **stand** -- how far the robot drifted and how far its pose moved.

Distributions are drawn as ECDFs rather than histograms: several runs overlay
without occluding each other, and the median and tail can be read straight off
the curve. A missing measurement (NaN -- no footstep landed, the goal was never
reached, the ball was never touched) is dropped rather than counted as zero, and
every panel that drops any says so, so a run that scores well on three episodes
cannot look like a run that scores well on a hundred.

Colours are the ``dataviz`` reference palette's first three categorical slots
(the set validated for all-pairs use in both light and dark) and its chart
chrome. One run gets slot 1 and no legend -- the title names it.
"""

from __future__ import annotations

import csv
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: this writes files, it never opens a window.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

from mjlab_piplus.tasks.eval.command_book import COMMAND_TYPES  # noqa: E402

__all__ = ["PlotConfig", "main"]


@dataclass(frozen=True)
class Theme:
  """One mode of the dataviz reference palette.

  The dark column is the same hues stepped for the dark surface, not an
  inversion of the light one.
  """

  surface: str
  ink: str
  ink_secondary: str
  muted: str
  grid: str
  axis: str
  series: tuple[str, str, str]


LIGHT = Theme(
  surface="#fcfcfb",
  ink="#0b0b0b",
  ink_secondary="#52514e",
  muted="#898781",
  grid="#e1e0d9",
  axis="#c3c2b7",
  series=("#2a78d6", "#eb6834", "#1baf7a"),
)
DARK = Theme(
  surface="#1a1a19",
  ink="#ffffff",
  ink_secondary="#c3c2b7",
  muted="#898781",
  grid="#2c2c2a",
  axis="#383835",
  series=("#3987e5", "#d95926", "#199e70"),
)

# Three, because the scatter panels put every pair of series on screen at once
# and only the first three slots clear the colour-separation floors that way.
MAX_RUNS = len(LIGHT.series)

# Point sizes before scaling. Every piece of text in a figure is drawn at
# ``pt(role)``, so one number moves all of them together and their relative
# weights stay put -- a 1.5x title beside an unscaled 8pt annotation reads as a
# mistake rather than as emphasis.
BASE_FONT = {
  "suptitle": 13.0,
  "title": 10.5,
  "label": 9.0,
  "tick": 8.5,
  "legend": 9.5,
  "note": 7.5,
  "annotation": 8.0,
}
# Where the run legend goes: under the plot (its own band), in the plot's top
# left corner (costing no band but covering whatever is there), or nowhere.
LEGEND_BELOW = "below"
LEGEND_INSIDE = "inside"
LEGEND_OFF = "off"
LEGEND_PLACEMENTS = (LEGEND_BELOW, LEGEND_INSIDE, LEGEND_OFF)

DEFAULT_FONT_SCALE = 1.5
_font_scale = DEFAULT_FONT_SCALE


def set_font_scale(scale: float) -> None:
  """Set the multiplier applied to every point size (and to the title bands)."""
  global _font_scale
  if scale <= 0.0:
    raise ValueError(f"font scale must be positive, got {scale}.")
  _font_scale = scale


def pt(role: str) -> float:
  return BASE_FONT[role] * _font_scale


def inches(base: float) -> float:
  """A layout band in inches, scaled with the text it has to hold."""
  return base * _font_scale


@dataclass
class Run:
  label: str
  path: Path
  tables: dict[str, dict[str, np.ndarray]]

  def has(self, command_type: str, *columns: str) -> bool:
    table = self.tables.get(command_type)
    return table is not None and all(column in table for column in columns)

  def col(self, command_type: str, column: str) -> np.ndarray:
    return self.tables[command_type][column]


##
# Loading.
##


def load_run(path: Path, label: str) -> Run:
  tables: dict[str, dict[str, np.ndarray]] = {}
  for command_type in COMMAND_TYPES:
    csv_path = path / f"episodes_{command_type}.csv"
    if not csv_path.exists():
      continue
    with csv_path.open() as handle:
      rows = list(csv.DictReader(handle))
    if not rows:
      continue
    tables[command_type] = {
      key: np.array([float(row[key]) for row in rows]) for key in rows[0]
    }
  if not tables:
    raise SystemExit(f"No episodes_*.csv under {path}. Is that an eval output dir?")
  return Run(label=label, path=path, tables=tables)


##
# Chart chrome.
##


def style_axes(ax, theme: Theme, title: str, xlabel: str, ylabel: str) -> None:
  """Hairline grid, two spines, muted axis ink. Everything recessive but the data."""
  ax.set_facecolor(theme.surface)
  ax.set_title(title, color=theme.ink, fontsize=pt("title"), pad=8)
  ax.set_xlabel(xlabel, color=theme.ink_secondary, fontsize=pt("label"))
  ax.set_ylabel(ylabel, color=theme.ink_secondary, fontsize=pt("label"))
  ax.grid(True, color=theme.grid, linewidth=0.8, linestyle="-")
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(theme.axis)
    ax.spines[side].set_linewidth(0.8)
  ax.tick_params(colors=theme.muted, labelsize=pt("tick"), length=3, width=0.8)


def note_coverage(ax, theme: Theme, notes: list[str]) -> None:
  """Say how many episodes actually produced a value, when any did not.

  A dropped NaN is an episode that never made the measurement; without this the
  reader cannot tell three good episodes from a hundred.
  """
  if not notes:
    return
  ax.text(
    0.98,
    0.04,
    "\n".join(notes),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=pt("note"),
    color=theme.muted,
    linespacing=1.4,
  )


def ecdf_panel(
  ax,
  theme: Theme,
  runs: list[Run],
  command_type: str,
  column: str,
  title: str,
  xlabel: str,
  scale: float = 1.0,
) -> None:
  """Fraction of episodes at or below x, one step line per run."""
  notes: list[str] = []
  for index, run in enumerate(runs):
    if not run.has(command_type, column):
      continue
    values = run.col(command_type, column) * scale
    finite = values[np.isfinite(values)]
    prefix = f"{run.label}: " if len(runs) > 1 else ""
    if finite.size == 0:
      notes.append(f"{prefix}never measured")
      continue
    if finite.size < values.size:
      notes.append(f"{prefix}{finite.size}/{values.size} episodes")
    ordered = np.sort(finite)
    fraction = np.arange(1, ordered.size + 1) / ordered.size
    ax.step(
      np.concatenate([[ordered[0]], ordered]),
      np.concatenate([[0.0], fraction]),
      where="post",
      color=theme.series[index],
      linewidth=1.8,
      label=run.label,
      solid_capstyle="round",
    )
  ax.set_ylim(0.0, 1.02)
  style_axes(ax, theme, title, xlabel, "episodes ≤ x")
  note_coverage(ax, theme, notes)


def scatter_panel(
  ax,
  theme: Theme,
  runs: list[Run],
  command_type: str,
  x_column: str,
  y_column: str,
  title: str,
  xlabel: str,
  ylabel: str,
  identity: bool = False,
  identity_label: str | None = None,
  x_scale: float = 1.0,
  y_scale: float = 1.0,
) -> None:
  """One marker per episode, with an optional ``y = x`` reference."""
  notes: list[str] = []
  limits: list[float] = []
  for index, run in enumerate(runs):
    if not run.has(command_type, x_column, y_column):
      continue
    x = run.col(command_type, x_column) * x_scale
    y = run.col(command_type, y_column) * y_scale
    keep = np.isfinite(x) & np.isfinite(y)
    prefix = f"{run.label}: " if len(runs) > 1 else ""
    if not keep.any():
      notes.append(f"{prefix}never measured")
      continue
    if keep.sum() < x.size:
      notes.append(f"{prefix}{int(keep.sum())}/{x.size} episodes")
    limits += [float(x[keep].min()), float(x[keep].max())]
    limits += [float(y[keep].min()), float(y[keep].max())]
    ax.scatter(
      x[keep],
      y[keep],
      s=34,
      color=theme.series[index],
      alpha=0.3,
      # A surface-coloured ring rather than a dark outline: it separates
      # overlapping markers without drawing a border around each one.
      edgecolors=theme.surface,
      linewidths=0.7,
      label=run.label,
      zorder=3,
    )
  if identity and limits:
    low, high = min(limits), max(limits)
    pad = 0.04 * (high - low or 1.0)
    line = np.array([low - pad, high + pad])
    ax.plot(line, line, color=theme.axis, linewidth=1.0, zorder=2)
    if identity_label:
      ax.annotate(
        identity_label,
        xy=(line[1], line[1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=pt("annotation"),
        color=theme.muted,
      )
  style_axes(ax, theme, title, xlabel, ylabel)
  note_coverage(ax, theme, notes)


def rate_panel(
  ax,
  theme: Theme,
  runs: list[Run],
  entries: list[tuple[str, str, str]],
  title: str,
  xlabel: str = "fraction of episodes",
) -> None:
  """Horizontal bars for rate-shaped columns.

  ``entries`` are ``(command_type, column, label)``. The value is the point
  here, so every bar is direct-labelled, outside its end where it cannot be
  clipped by a short bar.
  """
  labels = [label for _, _, label in entries]
  positions = np.arange(len(entries), dtype=float)
  group = 0.74
  height = group / max(len(runs), 1)
  for index, run in enumerate(runs):
    values = []
    for command_type, column, _ in entries:
      if run.has(command_type, column):
        column_values = run.col(command_type, column)
        finite = column_values[np.isfinite(column_values)]
        values.append(float(finite.mean()) if finite.size else math.nan)
      else:
        values.append(math.nan)
    offset = (index - (len(runs) - 1) / 2) * height
    # 0.86 of the slot leaves a surface gap between adjacent bars.
    bars = ax.barh(
      positions + offset,
      values,
      height=height * 0.86,
      color=theme.series[index],
      label=run.label,
      zorder=3,
    )
    for bar, value in zip(bars, values, strict=True):
      # A NaN here means the rate has no denominator -- no kick happened, so
      # "was it the intended foot" has nothing to average. Say so rather than
      # leaving a blank row that reads as a missing bar.
      finite = math.isfinite(value)
      ax.annotate(
        f"{value:.0%}" if finite else "not measured",
        xy=(value if finite else 0.0, bar.get_y() + bar.get_height() / 2),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=pt("annotation"),
        color=theme.ink_secondary if finite else theme.muted,
      )
  ax.set_yticks(positions)
  ax.set_yticklabels(labels)
  ax.set_ylim(positions[0] - 0.6, positions[-1] + 0.6)
  ax.invert_yaxis()
  ax.set_xlim(0.0, 1.12)
  ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
  style_axes(ax, theme, title, xlabel, "")
  ax.grid(axis="y", visible=False)


def wrap_title(text: str, width_in: float, size: float) -> str:
  """Break a title that is wider than its figure onto a second line.

  Wrapped rather than shrunk: the point size is a deliberate setting, and a
  title that quietly gets smaller the longer it is undoes it. Split at the
  ``figure: panel`` boundary where there is one, since that is where the title
  already divides in meaning.
  """
  # ~0.5 em per character for a mixed-case system sans; the exact renderer
  # extent would need a canvas that does not exist until the first draw.
  available = width_in - 0.3
  if len(text) * size * 0.5 / 72 <= available:
    return text
  if ": " in text:
    head, tail = text.split(": ", 1)
    return f"{head}:\n{tail}"
  return textwrap.fill(text, width=max(8, int(available * 144 / size)))


def title_lines(subject: str | None) -> int:
  return 0 if subject is None else subject.count("\n") + 1


def title_band(subject: str | None) -> float:
  """Inches of canvas a title block needs, added to the plot height.

  Only ever used to size the figure, so the plot area stays the same whether
  the title takes one line or two; ``tight_layout`` is what actually keeps the
  plot clear of the title.
  """
  if subject is None:
    return inches(0.10)
  return inches(0.05) + title_lines(subject) * pt("suptitle") * 1.15 / 72


def finish(
  fig,
  theme: Theme,
  subject: str | None,
  stem: Path,
  formats: tuple[str, ...],
  dpi: int,
  legend: str,
) -> list[Path]:
  """Title, shared legend, save once per format.

  The bands are sized in inches rather than in figure fractions, because the
  same code lays out a single-panel figure four inches tall and a six-panel
  sheet twice that: a fraction that leaves room for the title on one crushes
  the plot on the other. They scale with the font, so larger text gets more
  room rather than colliding with the plot.
  """
  height = fig.get_figheight()
  if subject is not None:
    # Anchored by its top, so a wrapped title grows downward into the band the
    # caller already sized for it.
    fig.suptitle(
      subject,
      color=theme.ink,
      fontsize=pt("suptitle"),
      va="top",
      y=1 - inches(0.05) / height,
    )

  # A legend for two or more series, never for one: with a single run the title
  # already names it and a one-entry legend is noise. Under ``--bare`` there is
  # no title to name it, so the legend stays whatever the run count.
  bottom = inches(0.06) / height
  handles, labels = [], []
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
      if label not in labels:
        handles.append(handle)
        labels.append(label)
  if legend != LEGEND_OFF and (len(handles) > 1 or (subject is None and handles)):
    if legend == LEGEND_INSIDE:
      # In the first panel only -- one legend per figure, not one per cell --
      # and over a surface-coloured patch, so it stays readable where it lands
      # on data. Stacked rather than in a row: a corner is taller than wide.
      box = fig.axes[0].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor=theme.surface,
        edgecolor="none",
        framealpha=0.85,
        borderpad=0.5,
        labelspacing=0.4,
        fontsize=pt("legend"),
      )
      box.set_zorder(5)
    else:
      box = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, inches(0.02) / height),
        ncol=len(labels),
        frameon=False,
        fontsize=pt("legend"),
      )
      # The band the plot gives up for it; an inside legend costs nothing here.
      bottom = inches(0.36) / height
    for text in box.get_texts():
      text.set_color(theme.ink_secondary)

  # The rect reserves the legend band only. ``tight_layout`` measures the
  # suptitle itself, so reserving a band for it here as well would double-count
  # it and leave a gap of one title's height between the title and the plot --
  # ``title_band`` is for sizing the canvas, not for carving the rect.
  #
  # ``pad`` is the figure-edge margin and must stay wide enough for a long
  # rotated y-label; ``h_pad`` is the gap between stacked rows, where that same
  # label would otherwise run into the panel above.
  fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0), pad=1.0, w_pad=1.0, h_pad=1.4)
  stem.parent.mkdir(parents=True, exist_ok=True)
  written = []
  for suffix in formats:
    path = stem.with_suffix(f".{suffix}")
    # dpi is ignored by the vector backend; it only sizes the raster output.
    fig.savefig(path, dpi=dpi, facecolor=theme.surface)
    written.append(path)
  plt.close(fig)
  return written


@dataclass(frozen=True)
class Panel:
  """One plot: a file of its own, and one cell of the optional contact sheet."""

  slug: str
  """File-name suffix -- ``<figure>_<slug>.pdf``."""
  title: str
  draw: Callable[[object], None]
  """Draws the panel into the axes it is handed."""


@dataclass(frozen=True)
class Figure:
  name: str
  title: str
  grid: tuple[int, int]
  """Rows and columns of the contact sheet. Ignored for the individual files."""
  panels: list[Panel]


@dataclass(frozen=True)
class Output:
  formats: tuple[str, ...]
  dpi: int
  bare: bool
  combined: bool
  legend: str


# Roughly a half-page figure: wide enough for an axis label under a date-free
# numeric axis, tall enough that an ECDF is not a flat smear. Slightly wider
# than tall, so a panel dropped into a two-column document does not eat the
# page.
PANEL_SIZE = (5.6, 3.7)


def subject_for(
  title: str, runs: list[Run], width_in: float, output: Output
) -> str | None:
  """The figure title, wrapped to the width it will be drawn at.

  ``None`` under ``--bare``: there is no title, and the caller sizes the
  figure without a title band.
  """
  if output.bare:
    return None
  text = f"{title} — {runs[0].label}" if len(runs) == 1 else title
  return wrap_title(text, width_in, pt("suptitle"))


def render(
  spec: Figure, theme: Theme, runs: list[Run], out: Path, output: Output
) -> list[Path]:
  """One file per panel, plus the contact sheet when asked for."""
  written: list[Path] = []
  width, plot_height = PANEL_SIZE
  for panel in spec.panels:
    subject = subject_for(f"{spec.title}: {panel.title}", runs, width, output)
    # The title band is added to the plot height rather than taken out of it,
    # so a title that wraps onto a second line grows the canvas instead of
    # squeezing the plot.
    fig, ax = plt.subplots(figsize=(width, plot_height + title_band(subject)))
    fig.patch.set_facecolor(theme.surface)
    panel.draw(ax)
    # The figure title names this panel (or, under --bare, the document's
    # caption does); the axes heading would just say it twice. Hidden rather
    # than emptied: an empty title still reserves its own font height plus
    # ``pad`` in the layout, which reappears as a gap under the figure title.
    ax.set_title("")
    ax.title.set_visible(False)
    written += finish(
      fig,
      theme,
      subject,
      out / f"{spec.name}_{panel.slug}",
      output.formats,
      output.dpi,
      output.legend,
    )
  if output.combined:
    rows, cols = spec.grid
    sheet_width = width * cols * 0.82
    subject = subject_for(spec.title, runs, sheet_width, output)
    fig, axes = plt.subplots(
      rows,
      cols,
      figsize=(sheet_width, plot_height * rows * 0.95 + title_band(subject)),
    )
    fig.patch.set_facecolor(theme.surface)
    flat = np.atleast_1d(axes).reshape(-1)
    for index, ax in enumerate(flat):
      if index < len(spec.panels):
        spec.panels[index].draw(ax)
      else:
        ax.set_visible(False)
    written += finish(
      fig,
      theme,
      subject,
      out / spec.name,
      output.formats,
      output.dpi,
      output.legend,
    )
  return written


##
# One figure per command type.
##


def build_velocity(theme: Theme, runs: list[Run]) -> Figure:
  panels: list[Panel] = []
  axes_spec = (
    ("cmd_vel_x", "mean_vel_x", "err_vel_x", "forward", "m/s", "forward"),
    ("cmd_vel_y", "mean_vel_y", "err_vel_y", "lateral", "m/s", "lateral"),
    ("cmd_yaw_rate", "mean_yaw_rate", "err_yaw_rate", "yaw rate", "rad/s", "yaw_rate"),
  )
  for cmd, achieved, _error, name, unit, slug in axes_spec:
    panels.append(
      Panel(
        f"{slug}_commanded_vs_achieved",
        f"{name}: commanded vs achieved",
        partial(
          scatter_panel,
          theme=theme,
          runs=runs,
          command_type="velocity",
          x_column=cmd,
          y_column=achieved,
          title=f"{name}: commanded vs achieved",
          xlabel=f"commanded ({unit})",
          ylabel=f"achieved ({unit})",
          identity=True,
          # Labelled in every panel, not just the first: each one is also a
          # file of its own, where "the panel to my left explains this line" is
          # not available.
          identity_label="y = x",
        ),
      )
    )
  for _cmd, _achieved, error, name, unit, slug in axes_spec:
    panels.append(
      Panel(
        f"{slug}_error",
        f"{name}: tracking error",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="velocity",
          column=error,
          title=f"{name}: tracking error",
          xlabel=f"mean |error| ({unit})",
        ),
      )
    )
  return Figure("velocity", "Velocity tracking", (2, 3), panels)


def build_goal_pose(theme: Theme, runs: list[Run]) -> Figure:
  return Figure(
    "goal_pose",
    "Goal pose",
    (1, 3),
    [
      Panel(
        "time_vs_distance",
        "Time to goal vs distance",
        partial(
          scatter_panel,
          theme=theme,
          runs=runs,
          command_type="goal_pose",
          x_column="initial_dist",
          y_column="time_to_goal",
          title="Time to goal vs distance",
          xlabel="initial distance (m)",
          ylabel="time to goal pose (s)",
        ),
      ),
      Panel(
        "final_distance",
        "Distance left at the end",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="goal_pose",
          column="final_dist",
          title="Distance left at the end",
          xlabel="final distance to goal (m)",
        ),
      ),
      Panel(
        "outcomes",
        "Outcomes",
        partial(
          rate_panel,
          theme=theme,
          runs=runs,
          entries=[
            ("goal_pose", "reached_position", "reached position"),
            ("goal_pose", "reached_goal", "reached full pose"),
            ("goal_pose", "fell", "fell"),
          ],
          title="Outcomes",
        ),
      ),
    ],
  )


def build_path(theme: Theme, runs: list[Run]) -> Figure:
  return Figure(
    "path",
    "Path following",
    (1, 3),
    [
      Panel(
        "path_distance",
        "Distance to the path",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="path",
          column="mean_path_dist",
          title="Distance to the path",
          xlabel="mean cross-track distance (m)",
        ),
      ),
      Panel(
        "worst_excursion",
        "Worst excursion",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="path",
          column="max_path_dist",
          title="Worst excursion",
          xlabel="max cross-track distance (m)",
        ),
      ),
      Panel(
        "time_vs_distance",
        "Time to goal vs distance",
        partial(
          scatter_panel,
          theme=theme,
          runs=runs,
          command_type="path",
          x_column="initial_dist_to_goal",
          y_column="time_to_goal",
          title="Time to goal vs distance",
          xlabel="initial distance to goal (m)",
          ylabel="time to goal (s)",
        ),
      ),
    ],
  )


def build_footstep(theme: Theme, runs: list[Run]) -> Figure:
  return Figure(
    "footstep",
    "Footstep placement",
    (2, 2),
    [
      Panel(
        "placement_error",
        "Placement error",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="footstep",
          column="step_err_xy",
          title="Placement error",
          xlabel="mean |xy error| (m)",
        ),
      ),
      Panel(
        "yaw_error",
        "Foot orientation error",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="footstep",
          column="step_err_yaw",
          title="Foot orientation error",
          xlabel="mean |yaw error| (deg)",
          scale=180.0 / math.pi,
        ),
      ),
      # The two errors against each other, one point per episode: says whether
      # a run that puts the foot in the wrong place is the same run that puts
      # it at the wrong angle, which neither distribution alone can show.
      Panel(
        "linear_vs_angular_error",
        "Linear vs angular error",
        partial(
          scatter_panel,
          theme=theme,
          runs=runs,
          command_type="footstep",
          x_column="step_err_xy",
          y_column="step_err_yaw",
          title="Linear vs angular error",
          xlabel="mean |xy error| (m)",
          ylabel="mean |yaw error| (deg)",
          y_scale=180.0 / math.pi,
        ),
      ),
      # Not "error vs footsteps taken": the schedule is read off the gait clock,
      # so every episode of a given length lands the same number of footsteps
      # and that scatter is a vertical line. The worst landing is the
      # informative second cut.
      Panel(
        "worst_landing",
        "Worst landing",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="footstep",
          column="step_err_xy_max",
          title="Worst landing",
          xlabel="max |xy error| (m)",
        ),
      ),
    ],
  )


def build_kick(theme: Theme, runs: list[Run]) -> Figure:
  return Figure(
    "kick",
    "Kick",
    (1, 3),
    [
      Panel(
        "speed_vs_commanded",
        "Commanded vs achieved speed",
        partial(
          scatter_panel,
          theme=theme,
          runs=runs,
          command_type="kick",
          x_column="cmd_strength",
          y_column="kick_speed",
          title="Commanded vs achieved speed",
          xlabel="commanded strength (m/s)",
          ylabel="ball peak speed (m/s)",
          identity=True,
          identity_label="y = x",
        ),
      ),
      Panel(
        "aim_error",
        "Aim error",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="kick",
          column="kick_aim_error",
          title="Aim error",
          xlabel="aim error (deg)",
          scale=180.0 / math.pi,
        ),
      ),
      Panel(
        "outcomes",
        "Outcomes",
        partial(
          rate_panel,
          theme=theme,
          runs=runs,
          entries=[
            ("kick", "kicked", "kicked at all"),
            # Spelled out because the denominator differs from the bar above
            # it: the foot is only recorded where a kick actually happened.
            ("kick", "kick_right_foot", "intended foot (of kicks)"),
            ("kick", "fell", "fell"),
          ],
          title="Outcomes",
        ),
      ),
    ],
  )


def build_stand(theme: Theme, runs: list[Run]) -> Figure:
  return Figure(
    "stand",
    "Standing",
    (1, 3),
    [
      Panel(
        "drift",
        "Base drift",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="stand",
          column="drift",
          title="Base drift",
          xlabel="drift from spawn (m)",
        ),
      ),
      Panel(
        "yaw_drift",
        "Heading drift",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="stand",
          column="yaw_drift",
          title="Heading drift",
          xlabel="|yaw drift| (deg)",
          scale=180.0 / math.pi,
        ),
      ),
      Panel(
        "posture",
        "Posture",
        partial(
          ecdf_panel,
          theme=theme,
          runs=runs,
          command_type="stand",
          column="posture_rms",
          title="Posture",
          xlabel="RMS joint deviation (rad)",
        ),
      ),
    ],
  )


# Only some types have a notion of "the task was completed"; velocity, footstep
# and stand are scored by how well, not whether.
SUCCESS_COLUMNS: dict[str, tuple[str, str]] = {
  "path": ("reached_goal", "path: reached goal"),
  "goal_pose": ("reached_goal", "goal pose: reached"),
  "kick": ("kicked", "kick: struck the ball"),
}


def build_overview(theme: Theme, runs: list[Run], types: list[str]) -> Figure:
  panels = [
    Panel(
      "fall_rate",
      "Fall rate by command type",
      partial(
        rate_panel,
        theme=theme,
        runs=runs,
        entries=[(name, "fell", name) for name in types],
        title="Fall rate by command type",
      ),
    )
  ]
  successes = [
    (name, *SUCCESS_COLUMNS[name]) for name in types if name in SUCCESS_COLUMNS
  ]
  if successes:
    panels.append(
      Panel(
        "task_success",
        "Task success, where defined",
        partial(
          rate_panel,
          theme=theme,
          runs=runs,
          entries=successes,
          title="Task success, where defined",
        ),
      )
    )
  return Figure("overview", "Evaluation overview", (1, 2), panels)


BUILDERS = {
  "velocity": build_velocity,
  "path": build_path,
  "goal_pose": build_goal_pose,
  "footstep": build_footstep,
  "kick": build_kick,
  "stand": build_stand,
}

FORMATS = {"pdf": ("pdf",), "png": ("png",), "both": ("pdf", "png")}


@dataclass(frozen=True)
class PlotConfig:
  runs: tuple[Path, ...]
  """Evaluation output directories (the ones holding episodes_*.csv). Two or
  three overlay for comparison."""
  labels: tuple[str, ...] | None = None
  """Series names, in the same order as ``runs``. Defaults to the directory
  names."""
  out: Path | None = None
  """Where the files go. Defaults to ``<first run>/plots``."""
  types: tuple[str, ...] = COMMAND_TYPES
  """Which command types to plot. Types with no CSV are skipped."""
  format: str = "pdf"
  """``pdf`` (vector, for a document), ``png``, or ``both``."""
  combined: bool = False
  """Also write the multi-panel contact sheet per command type. Off by default:
  the individual files are the deliverable, the sheet is for reviewing them."""
  bare: bool = False
  """Drop the title, leaving the plot alone. For a figure whose document
  supplies the caption. The legend stays unless ``--legend off`` drops it."""
  legend: str = LEGEND_BELOW
  """Where the legend naming the runs goes: ``below`` the plot in its own band,
  ``inside`` the plot's top-left corner (no band, so the plot gets that space
  back, at the cost of covering whatever sits in that corner), or ``off``.
  ``off`` leaves two or more runs distinguishable by colour alone, so use it
  only where a caption or a shared legend elsewhere names them."""
  font_scale: float = DEFAULT_FONT_SCALE
  """Multiplier on every point size -- titles, axis labels, ticks, legend and
  annotations alike. The title and legend bands scale with it, so the layout
  follows the text instead of colliding with it."""
  theme: str = "light"
  """``light`` or ``dark``. The dark palette is its own set of steps for the
  dark surface, not an inversion."""
  dpi: int = 160
  """Raster resolution. Ignored for PDF, which is vector."""


def main(cfg: PlotConfig) -> None:
  if not cfg.runs:
    raise SystemExit("Pass at least one evaluation directory with --runs.")
  if len(cfg.runs) > MAX_RUNS:
    raise SystemExit(
      f"{len(cfg.runs)} runs given; these panels put every pair of series on "
      f"screen at once and stay readable for at most {MAX_RUNS}. Plot them in "
      "groups instead -- cycling more colours would make pairs that nobody "
      "with a colour-vision deficiency can tell apart."
    )
  if cfg.labels is not None and len(cfg.labels) != len(cfg.runs):
    raise SystemExit(
      f"{len(cfg.labels)} labels for {len(cfg.runs)} runs; they must line up."
    )
  unknown = [name for name in cfg.types if name not in COMMAND_TYPES]
  if unknown:
    raise SystemExit(f"Unknown command type(s) {unknown}. Known: {COMMAND_TYPES}.")

  theme = {"light": LIGHT, "dark": DARK}.get(cfg.theme.lower())
  if theme is None:
    raise SystemExit(f"--theme must be 'light' or 'dark', got '{cfg.theme}'.")
  formats = FORMATS.get(cfg.format.lower())
  if formats is None:
    raise SystemExit(f"--format must be one of {sorted(FORMATS)}, got '{cfg.format}'.")
  if cfg.font_scale <= 0.0:
    raise SystemExit(f"--font-scale must be positive, got {cfg.font_scale}.")
  legend = cfg.legend.lower()
  if legend not in LEGEND_PLACEMENTS:
    raise SystemExit(
      f"--legend must be one of {list(LEGEND_PLACEMENTS)}, got '{cfg.legend}'."
    )

  labels = cfg.labels or tuple(path.resolve().name for path in cfg.runs)
  runs = [load_run(path, label) for path, label in zip(cfg.runs, labels, strict=True)]
  out = cfg.out or (cfg.runs[0] / "plots")
  set_font_scale(cfg.font_scale)
  output = Output(
    formats=formats,
    dpi=cfg.dpi,
    bare=cfg.bare,
    combined=cfg.combined,
    legend=legend,
  )

  present = [name for name in cfg.types if any(name in run.tables for run in runs)]
  missing = [name for name in cfg.types if name not in present]
  if missing:
    print(f"[INFO] No CSV for {missing}; skipping.")
  if not present:
    raise SystemExit("None of the requested command types has a CSV.")

  written = render(build_overview(theme, runs, present), theme, runs, out, output)
  for name in present:
    written += render(BUILDERS[name](theme, runs), theme, runs, out, output)
  for path in written:
    print(f"  -> {path}")


if __name__ == "__main__":
  main(tyro.cli(PlotConfig))
