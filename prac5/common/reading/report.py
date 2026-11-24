import matplotlib.pyplot as plt

from .drowsiness import CALIBRATION_S, STEP_S, WINDOWS
from .overlay import NAMES

STATE_COLORS = {"alerta": "#1baf7a", "poco atento": "#2a78d6", "somnoliento": "#eb6834"}
EMOTION_COLORS = {"angry": "#d62728", "disgust": "#8c564b", "fear": "#9467bd", "happy": "#e8b40c", "sad": "#1f77b4",
                  "surprise": "#17becf", "neutral": "#c9c9c9"}


def minutes(axis):
    axis.xaxis.set_major_locator(plt.MultipleLocator(60))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda s, _: f"{int(s // 60)}:{int(s % 60):02d}"))


def plot_drowsiness(ax, states):
    full = CALIBRATION_S + WINDOWS[-1]
    ax.text(CALIBRATION_S / 2, 0.03, "calibrando", ha="center", color="0.45", fontsize=9)
    ax.axvline(full, color="0.6", ls="--", lw=1)
    ax.text(full + 2, 1.0, f"ventana completa ({WINDOWS[-1]} s)", color="0.45", fontsize=9, va="top")
    for state, color in STATE_COLORS.items():
        if len(states):
            ax.plot(states.index, states[state], color=color, lw=2.2, marker="o", ms=3)
            ax.annotate(state, (states.index[-1], states[state].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=color, fontweight="bold")
    ax.set(ylim=(0, 1.05), ylabel="probabilidad",
           title=f"Somnolencia (cada {STEP_S} s; ventana de {WINDOWS[0]} a {WINDOWS[-1]} s)")
    ax.grid(axis="y", alpha=0.3)


def plot_emotions(ax, shares, duration):
    dominant, share = shares.idxmax(axis=1), shares.max(axis=1)
    ax.bar(dominant.index, share, width=STEP_S, align="edge", color=[EMOTION_COLORS[e] for e in dominant], lw=0)
    ax.set(ylim=(0, 1), xlim=(0, duration), ylabel="% de caras", xlabel="tiempo de lectura (min)",
           title=f"Emoción predominante por tramos de {STEP_S} s")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    shown = [e for e in EMOTION_COLORS if e in set(dominant)]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=EMOTION_COLORS[e]) for e in shown], [NAMES[e] for e in shown],
              loc="upper left", frameon=False, ncol=len(EMOTION_COLORS), bbox_to_anchor=(0, -0.35))


def plot_report(states, shares, duration):
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    for axis in (ax, ax_e):
        axis.axvspan(0, CALIBRATION_S, color="0.93", lw=0)
        axis.spines[["top", "right"]].set_visible(False)
    plot_drowsiness(ax, states)
    plot_emotions(ax_e, shares, duration)
    minutes(ax_e)
    fig.tight_layout()
    return fig
