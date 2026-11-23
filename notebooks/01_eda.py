import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CSV_SEP, EDA_FIGURES_DIR, RAW_DATA_PATH, TARGET_COL


def main() -> None:
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH, sep=CSV_SEP)
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    # --- Gráfica 1: Proporción de clases y valores 'unknown' ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    y_counts = df[TARGET_COL].value_counts()
    bars = axes[0].bar(y_counts.index, y_counts.values, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Proporción de la clase objetivo (y)")
    axes[0].set_xlabel("Contrató depósito a plazo (y)")
    axes[0].set_ylabel("Cantidad de clientes")
    for bar, val in zip(bars, y_counts.values):
        pct = val / len(df) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 60,
            f"{val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylim(0, y_counts.max() * 1.18)

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    unk_counts = pd.Series({c: (df[c] == "unknown").sum() for c in obj_cols}).sort_values()
    unk_counts = unk_counts[unk_counts > 0]
    axes[1].barh(unk_counts.index, unk_counts.values, color="#7A68AE")
    axes[1].set_title("Valores 'unknown' por variable (0 nulos NaN)")
    axes[1].set_xlabel("Cantidad de registros con 'unknown'")
    axes[1].set_ylabel("Variable categórica")
    for i, v in enumerate(unk_counts.values):
        axes[1].text(v + 50, i, f"{v} ({v / len(df) * 100:.1f}%)", va="center", fontsize=9)
    axes[1].set_xlim(0, unk_counts.max() * 1.22)

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "eda_clases_faltantes.png", dpi=300)
    plt.close()

    # --- Gráfica 2: Distribuciones de variables numéricas clave ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_cols = ["age", "balance", "duration", "campaign"]
    titles = [
        "Edad (age)",
        "Saldo anual promedio (balance)",
        "Duración llamada en seg. (duration)",
        "Contactos en campaña (campaign)",
    ]

    for ax, col, title in zip(axes.ravel(), plot_cols, titles):
        sns.histplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            bins=30,
            ax=ax,
            element="step",
            common_norm=False,
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "eda_distribuciones.png", dpi=300)
    plt.close()

    # --- Gráfica 3: Matriz de correlación lineal ---
    plt.figure(figsize=(8, 6))
    num_df = df.select_dtypes(include=["number"]).copy()
    num_df["y_num"] = (df[TARGET_COL].str.lower() == "yes").astype(int)
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlación lineal entre variables numéricas y y_num")
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "eda_correlaciones.png", dpi=300)
    plt.close()

    print("Gráficas EDA guardadas en:", EDA_FIGURES_DIR)


if __name__ == "__main__":
    main()