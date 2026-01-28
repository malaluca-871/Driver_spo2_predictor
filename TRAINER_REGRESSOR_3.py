#!/usr/bin/env python3
"""
Training script per modello di regressione "SpO2" usando feature rPPG
(derivate da YCgCr + altre).

Struttura CSV:
- csv_dir/
    PURE_01/
        features_01-01.csv
        features_01-02.csv
        ...
    PURE_02/
        ...
    PD_AFH1/
        ...
    ...

Dove:
  - group_id   = nome della sottocartella (persona reale)
  - subject_id = nome del file (sessione / registrazione)

Lo split train/test e il GroupKFold sono fatti per group_id
(per evitare leakage tra sessioni della stessa persona).

NOVITÀ:
- Le predizioni sul TEST sono calcolate solo ogni ~1 secondo
  (sottocampionando le finestre del CSV, che sono ogni frame ~1/30 s).
- Le predizioni SpO2 sul TEST vengono arrotondate all'intero
  (come il GT proveniente dal pulsossimetro).
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler

# --------------------- CONFIGURAZIONE ------------------------------------
TH_LOW = 95.0       # soglia SpO2 per "a rischio" (usata solo per plotting/analisi)
N_ESTIMATORS = 300  # alberi RandomForest
RANDOM_STATE = 42

# sottocampionamento temporale del TEST:
# intervallo (in secondi) tra due predizioni consecutive
TEST_PRED_STEP_SEC = 1.0
# ------------------------------------------------------------------------


def load_all_csv(csv_dir: str) -> pd.DataFrame:
    """
    Carica TUTTI i CSV (ricorsivamente) in csv_dir.

    Aggiunge:
      - 'group_id'   = nome della cartella subito sopra (persona)
      - 'subject_id' = nome del file senza estensione (sessione)

    Ritorna un unico DataFrame concatenato.
    """
    paths = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))

    if len(paths) == 0:
        raise RuntimeError(f"Nessun CSV trovato in {csv_dir}")

    dfs = []
    for p in paths:
        base = os.path.basename(p)

        # evita eventuali file riepilogo tipo "features_all_subjects.csv"
        if "all_subjects" in base.lower():
            print(f"[WARN] Skip file riepilogo: {p}")
            continue

        df = pd.read_csv(p)

        subject_id = os.path.splitext(base)[0]          # es. "features_01-01"
        group_id = os.path.basename(os.path.dirname(p))  # es. "PURE_01" o "PD_AFH1"

        df["subject_id"] = subject_id
        df["group_id"] = group_id
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Solo file riepilogo trovati, nessuna sessione valida.")

    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    return all_df


def build_dataset(df: pd.DataFrame):
    """
    Costruisce:
      - X: feature numeriche
      - y: target continuo (gt_spo2)
      - groups: group_id (persona) per GroupKFold / split
      - feature_cols: nomi colonne feature
      - df_clean: df filtrato (senza gt_spo2 NaN e outlier), indice riassegnato
    """
    df_clean = df.copy()
    df_clean = df_clean[~df_clean["gt_spo2"].isna()].reset_index(drop=True)

    # Filtra outlier evidentemente non fisiologici
    df_clean = df_clean[
        (df_clean["gt_spo2"] >= 90.0) & (df_clean["gt_spo2"] <= 100.0)
    ]

    # Target continuo
    y = df_clean["gt_spo2"].values.astype(float)

    # Gruppi = persona reale (cartella)
    if "group_id" not in df_clean.columns:
        raise RuntimeError("Manca la colonna 'group_id' nel DataFrame. "
                           "Controlla load_all_csv / struttura cartelle.")
    groups = df_clean["group_id"].values

    drop_cols = {"gt_spo2", "time_s_pred"}
    feature_cols = [
        c for c in df_clean.columns
        if c not in drop_cols
        and c not in {"subject_id", "group_id"}
        and np.issubdtype(df_clean[c].dtype, np.number)
    ]
    X = df_clean[feature_cols].values

    print("Feature usate:")
    for c in feature_cols:
        print(f"  - {c}")

    print(f"Totale campioni: {len(df_clean)}")
    print(f"Range SpO2: min={np.min(y):.2f}, max={np.max(y):.2f}, "
          f"mean={np.mean(y):.2f}, std={np.std(y):.2f}")

    return X, y, groups, feature_cols, df_clean


def stratified_subject_split(groups, y, test_size=0.3, random_state=42):
    """
    Splitta le PERSONE (group_id) in train/test cercando di mantenere
    la proporzione tra:
        - persone con almeno 1 campione < TH_LOW
        - persone sempre >= TH_LOW.

    groups: array di group_id (persona)
    y:     array target (SpO2)

    Ritorna: subj_train, subj_test (array di group_id)
    """
    subjects = np.unique(groups)
    subj_labels = []

    for s in subjects:
        mask = (groups == s)
        subj_has_risk = (y[mask] < TH_LOW).any()
        subj_labels.append(int(subj_has_risk))

    subj_labels = np.array(subj_labels)
    unique_classes, counts = np.unique(subj_labels, return_counts=True)

    # Caso 1: solo una classe di persone
    if len(unique_classes) < 2:
        print("⚠️ Solo una tipologia di persone (tutte con o senza episodi < soglia); "
              "uso train_test_split non stratificato.")
        subj_train, subj_test = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    # Caso 2: due classi ma una con meno di 2 persone → impossibile stratificare
    elif (counts < 2).any():
        print("⚠️ Una classe di persone ha meno di 2 soggetti; "
              "uso train_test_split non stratificato.")
        print("   Distribuzione persone per classe (0=solo >= soglia, 1=≥1 < soglia):")
        for cls, cnt in zip(unique_classes, counts):
            print(f"   Classe {cls}: {cnt} persone")
        subj_train, subj_test = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    # Caso 3: stratificazione possibile
    else:
        subj_train, subj_test = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=subj_labels
        )

    # Debug: distribuzione per persona
    print("\nDistribuzione persone (a livello di group_id):")
    all_subj = subjects
    for name, subset in [("TRAIN", subj_train), ("TEST", subj_test)]:
        mask = np.isin(all_subj, subset)
        labs = subj_labels[mask]
        n_risk = (labs == 1).sum()
        n_norm = (labs == 0).sum()
        print(f"  {name}: {len(subset)} persone -> "
              f"{n_risk} con ≥1 campione < {TH_LOW}, {n_norm} solo >= {TH_LOW}")

    return subj_train, subj_test


def downsample_test_mask_by_time(df_clean: pd.DataFrame,
                                 base_test_mask: np.ndarray,
                                 seconds: float = TEST_PRED_STEP_SEC) -> np.ndarray:
    """
    Crea una nuova mask booleana (stessa lunghezza di df_clean) che seleziona
    solo alcune righe del TEST, in modo da avere una predizione ogni ~`seconds`
    secondi per ciascuna SESSIONE (subject_id).

    Logica:
      - si lavora solo sulle righe con base_test_mask=True
      - per ogni (group_id, subject_id), si ordina per time_s_pred
      - si stima il dt medio tra campioni (es. ~1/30 s)
      - si calcola frames_per_sec ≈ 1/dt
      - step_frames = seconds * frames_per_sec  (es. ≈ 30)
      - si prende una riga ogni step_frames
    """
    idx_all_test = np.where(base_test_mask)[0]
    df_test = df_clean.iloc[idx_all_test].copy()

    if "time_s_pred" not in df_test.columns:
        raise RuntimeError("Manca la colonna 'time_s_pred' per il sottocampionamento temporale.")

    selected_global_idx = []

    # groupby per persona+sessione, per robustezza
    for (_, _), sub_df in df_test.groupby(["group_id", "subject_id"]):
        sub_df = sub_df.sort_values("time_s_pred")
        times = sub_df["time_s_pred"].values

        if len(times) < 2:
            # se c'è un solo punto, lo teniamo
            selected_global_idx.extend(sub_df.index.tolist())
            continue

        dt = np.diff(times)
        dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0

        if dt_med <= 0:
            step_frames = 1
        else:
            frames_per_sec = int(round(1.0 / dt_med))
            step_frames = max(1, int(round(seconds * frames_per_sec)))

        selected_idx_sub = sub_df.index[::step_frames]
        selected_global_idx.extend(selected_idx_sub.tolist())

    selected_global_idx = np.array(sorted(set(selected_global_idx)), dtype=int)

    new_mask = np.zeros_like(base_test_mask, dtype=bool)
    new_mask[selected_global_idx] = True

    print(f"\nSottocampionamento TEST: da {base_test_mask.sum()} a {new_mask.sum()} campioni "
          f"(step ≈ {seconds}s)")
    return new_mask


def _sanitize_filename(s: str) -> str:
    """Rende una stringa sicura per l'uso in un nome file."""
    s = str(s)
    s = re.sub(r"[^\w\-]+", "_", s)
    return s


def plot_spo2_time_series_for_test(df_clean, test_mask, y_test, y_test_pred, seed):
    """
    Per ogni SESSIONE nel test set (subject_id), plottiamo:
      - GT SpO2 nel tempo
      - predizione SpO2
      - linea orizzontale alla soglia TH_LOW

    NOTA: test_mask è già quello sottocampionato (una predizione circa ogni 1s).
    """
    df_test = df_clean.loc[test_mask].copy()
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_test_pred

    subjects_test = df_test["subject_id"].unique()

    print("\n=== Plot SpO2 (GT vs pred) nel tempo per sessioni nel test (seed={}) ===".format(seed))
    for subj in subjects_test:
        sub_df = df_test[df_test["subject_id"] == subj].copy()
        if "time_s_pred" not in sub_df.columns or "gt_spo2" not in sub_df.columns:
            print(f"  Sessione {subj}: mancano colonne time_s_pred o gt_spo2, plot SKIPPATO.")
            continue

        sub_df = sub_df.sort_values("time_s_pred")

        group_id = sub_df["group_id"].iloc[0] if "group_id" in sub_df.columns else "NA"
        print(f"  Plot sessione {subj} (persona {group_id}): {len(sub_df)} campioni nel test (seed={seed})")

        times = sub_df["time_s_pred"].values
        spo2_gt = sub_df["y_true"].values
        spo2_pred = sub_df["y_pred"].values

        plt.figure(figsize=(8, 4))
        plt.plot(times, spo2_gt, linestyle="-", label="GT SpO₂")
        plt.plot(times, spo2_pred, linestyle="--", marker="o", alpha=0.7, label="Pred SpO₂ (1s step)")
        plt.axhline(TH_LOW, linestyle=":", color="gray", label=f"Soglia {TH_LOW:.1f}%")

        plt.xlabel("Tempo [s]")
        plt.ylabel("SpO₂ [%]")
        plt.title(f"GT vs Pred SpO₂ - Sessione {subj} (persona {group_id}, seed={seed})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        subj_safe = _sanitize_filename(subj)
        fname = f"spo2_timeseries_regression_session_{subj_safe}_seed{seed}.png"
        plt.savefig(fname, dpi=200)
        plt.show()


def run_single_split(df, seed: int, do_plots_and_csv: bool = True):
    """
    Esegue un singolo run di training+valutazione con:
      - split persone train/test basato su 'seed'
      - GroupKFold sul train (per persona)
      - training finale e valutazione su test

    Modalità REGRESSIONE:
      - target: SpO2 continua
      - metriche: MAE, RMSE, R^2

    NOVITÀ:
      - Il TEST è sottocampionato a una finestra ogni ~1 secondo.
      - Le predizioni sul TEST sono arrotondate all'intero.
    """
    print("\n" + "=" * 60)
    print(f"=== RUN con random_state={seed} (REGRESSIONE SpO2) ===")
    print("=" * 60)

    # 2) Costruisci dataset
    X, y, groups, feature_cols, df_clean = build_dataset(df)

    # 3) Split train/test per persona (group_id)
    subj_train, subj_test = stratified_subject_split(
        groups, y, test_size=0.3, random_state=seed
    )

    base_train_mask = np.isin(groups, subj_train)
    base_test_mask = np.isin(groups, subj_test)

    # sottocampioniamo SOLO il test: il train resta denso per usare più dati
    test_mask = downsample_test_mask_by_time(df_clean, base_test_mask,
                                             seconds=TEST_PRED_STEP_SEC)
    train_mask = base_train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_train = groups[train_mask]
    groups_test = groups[test_mask]

    print(f"\nPersone train: {len(np.unique(groups_train))}, "
          f"persone test: {len(np.unique(groups_test))}")
    print(f"Campioni train: {len(y_train)}, campioni test (sottocampionati): {len(y_test)}")

    # 4) Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Cross-validation interna sul train (GroupKFold per persona)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
    print("\n=== Cross-val (GroupKFold) sul train ===")
    cv_scores = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_scaled, y_train, groups_train), 1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        reg = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=seed,
            n_jobs=-1
        )
        reg.fit(X_tr, y_tr)

        y_val_pred = reg.predict(X_val)

        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)

        cv_scores.append((mae, rmse, r2))
        print(f"Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}, R^2={r2:.3f}")

    if cv_scores:
        cv_scores = np.array(cv_scores)
        print("\nMedia CV (train):")
        print(f"  MAE={cv_scores[:, 0].mean():.3f}")
        print(f"  RMSE={cv_scores[:, 1].mean():.3f}")
        print(f"  R^2={cv_scores[:, 2].mean():.3f}")
    else:
        print("\nNessun fold di CV valido (improbabile per regressore).")

    # 6) Fit finale su tutto il train
    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1
    )
    reg.fit(X_train_scaled, y_train)

    # 7) Valutazione su test
    y_test_pred = reg.predict(X_test_scaled)

    # Arrotondamento delle predizioni all'intero, e GT portato a intero
    y_test_int = np.rint(y_test).astype(int)
    y_test_pred_int = np.rint(y_test_pred).astype(int)

    mae = mean_absolute_error(y_test_int, y_test_pred_int)
    rmse = np.sqrt(mean_squared_error(y_test_int, y_test_pred_int))
    r2 = r2_score(y_test_int, y_test_pred_int)

    print("\n=== Test set (persone held-out, pred ogni ~1s, valori interi) ===")
    print(f"MAE    = {mae:.3f}")
    print(f"RMSE   = {rmse:.3f}")
    print(f"R^2    = {r2:.3f}")

    # --- Plot SpO2 nel tempo per ogni sessione nel test ---
    if do_plots_and_csv:
        plot_spo2_time_series_for_test(
            df_clean=df_clean,
            test_mask=test_mask,
            y_test=y_test_int,
            y_test_pred=y_test_pred_int,
            seed=seed
        )

    # 8) Scatter plot GT vs Pred
    if do_plots_and_csv:
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test_int, y_test_pred_int, alpha=0.6)
        min_val = min(y_test_int.min(), y_test_pred_int.min())
        max_val = max(y_test_int.max(), y_test_pred_int.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
        plt.xlabel("GT SpO₂ (int)")
        plt.ylabel("Pred SpO₂ (int, 1s step)")
        plt.title(f"GT vs Pred SpO₂ (test set, seed={seed})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"scatter_gt_vs_pred_spo2_seed{seed}.png", dpi=200)
        plt.show()

    # 9) Feature importances (sul modello continuo, non arrotondato)
    if do_plots_and_csv:
        importances = reg.feature_importances_
        idx_sorted = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(importances)), importances[idx_sorted])
        plt.xticks(range(len(importances)),
                   [feature_cols[i] for i in idx_sorted],
                   rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(f"RandomForestRegressor feature importances (seed={seed})")
        plt.tight_layout()
        plt.savefig(f"feature_importances_spo2_regressor_seed{seed}.png", dpi=200)
        plt.show()

    # 10) Salva CSV con predizioni sul test (solo punti sottocampionati)
    if do_plots_and_csv:
        df_test_meta = df_clean.loc[test_mask, ["subject_id", "group_id", "time_s_pred"]].reset_index(drop=True)

        out_df = pd.DataFrame({
            "group_id": df_test_meta["group_id"],
            "subject_id": df_test_meta["subject_id"],
            "time_s_pred": df_test_meta["time_s_pred"],
            "spo2_true_int": y_test_int,
            "spo2_pred_int": y_test_pred_int,
        })
        out_df.to_csv(f"spo2_regressor_test_predictions_seed{seed}.csv", index=False)
        print(f"Saved spo2_regressor_test_predictions_seed{seed}.csv")

    return {
        "seed": seed,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_test_samples": int(len(y_test_int))
    }


def main(args):
    # 1) Carica CSV UNA sola volta
    df = load_all_csv(args.csv_dir)

    n_repeats = args.n_repeats

    if n_repeats == 1:
        _ = run_single_split(df, seed=RANDOM_STATE, do_plots_and_csv=True)
        return

    # Multi-run: più split con random_state diversi
    results = []
    for i in range(n_repeats):
        seed = RANDOM_STATE + i
        res = run_single_split(df, seed=seed, do_plots_and_csv=True)
        results.append(res)

    # Riassunto delle metriche sui vari split
    print("\n" + "#" * 60)
    print("RIASSUNTO SU PIÙ SPLIT (test set) - REGRESSIONE SpO2")
    print("#" * 60)

    res_df = pd.DataFrame(results)
    print(res_df[["seed", "mae", "rmse", "r2", "n_test_samples"]])

    res_df.to_csv("spo2_regressor_runs_metrics.csv", index=False)
    print("Saved spo2_regressor_runs_metrics.csv")

    def summary(col):
        vals = res_df[col].values
        return np.nanmean(vals), np.nanstd(vals)

    for metric in ["mae", "rmse", "r2"]:
        mean, std = summary(metric)
        print(f"{metric.upper()}: mean={mean:.3f}, std={std:.3f}")

    # GRAFICI DELLE PERFORMANCE PER OGNI RIPETIZIONE
    metrics = ["mae", "rmse", "r2"]
    plt.figure(figsize=(8, 5))
    for m in metrics:
        plt.plot(res_df["seed"], res_df[m], marker="o", label=m.upper())
    plt.xlabel("Random seed / run")
    plt.ylabel("Valore metrica")
    plt.title("Performance del modello di regressione per ripetizione (test set, 1s step)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_metrics_per_run_lineplot.png", dpi=200)
    plt.show()

    plt.figure(figsize=(6, 4))
    data_for_box = [res_df[m].dropna() for m in metrics]
    plt.boxplot(data_for_box, labels=[m.upper() for m in metrics])
    plt.ylabel("Valore metrica")
    plt.title("Distribuzione delle metriche sulle ripetizioni (test set, 1s step)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_metrics_boxplot_across_runs.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Cartella ROOT che contiene le sottocartelle dei soggetti (PURE_01, PD_..., ecc.)")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Numero di split train/test da ripetere con random_state diversi")
    args = parser.parse_args()
    main(args)
