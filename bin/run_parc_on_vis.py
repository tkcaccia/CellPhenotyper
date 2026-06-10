#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parc
from sklearn.metrics import silhouette_score


def parse_args():
    p = argparse.ArgumentParser(description="Run PARC on a 2D vis CSV and save clusters/plot.")
    p.add_argument("--vis-csv", required=True)
    p.add_argument("--cell-ids", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--prefix", default="parc_fullvis")
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--n-iter-leiden", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--plot-max", type=int, default=220000)
    p.add_argument("--silhouette-max", type=int, default=4000)
    return p.parse_args()


def renumber_by_size(labels):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    mapping = {old: i + 1 for i, old in enumerate(order)}
    return np.asarray([mapping[x] for x in labels], dtype=np.int32)


def read_ids(path, n):
    ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            ids.append(line.rstrip("\n"))
    if len(ids) != n:
        raise ValueError(f"cell ID count {len(ids)} does not match vis rows {n}")
    return ids


def palette(labels):
    unique = np.unique(labels)
    cmap = plt.get_cmap("turbo", max(len(unique), 2))
    colors = {cid: cmap(i / max(len(unique) - 1, 1)) for i, cid in enumerate(unique)}
    return colors


def plot_clusters(vis, labels, out_png, title, plot_max, seed):
    rng = np.random.default_rng(seed)
    n = vis.shape[0]
    if n > plot_max:
        idx = np.sort(rng.choice(n, size=plot_max, replace=False))
    else:
        idx = np.arange(n)
    colors = palette(labels)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for cid in np.unique(labels):
        ii = idx[labels[idx] == cid]
        if ii.size == 0:
            continue
        ax.scatter(vis[ii, 0], vis[ii, 1], s=0.4, c=[colors[cid]], alpha=0.70, linewidths=0, label=str(cid))
    ax.set_xlabel("KODAMA dimension 1")
    ax.set_ylabel("KODAMA dimension 2")
    ax.set_title(title)
    sizes = sorted([(cid, int(np.sum(labels == cid))) for cid in np.unique(labels)], key=lambda x: -x[1])
    if len(sizes) <= 15:
        ax.legend(title="cluster", markerscale=8, fontsize=6, title_fontsize=7, loc="upper right", frameon=False)
    else:
        text = "Top clusters\n" + "\n".join([f"{cid}: {count}" for cid, count in sizes[:12]])
        ax.text(0.99, 0.99, text, transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading vis from {args.vis_csv}", flush=True)
    t0 = time.time()
    vis = np.loadtxt(args.vis_csv, delimiter=",", dtype=np.float32)
    print(f"[INFO] Loaded vis shape={vis.shape} dtype={vis.dtype} elapsed={time.time() - t0:.1f}s", flush=True)

    t_ids = time.time()
    ids = read_ids(args.cell_ids, vis.shape[0])
    print(f"[INFO] Loaded cell ids elapsed={time.time() - t_ids:.1f}s", flush=True)

    run_name = f"{args.prefix}_knn{args.knn}_res{str(args.resolution).replace('.', 'p')}"
    print(f"[INFO] Running PARC {run_name}", flush=True)
    t_parc = time.time()
    model = parc.PARC(
        vis,
        knn=args.knn,
        resolution_parameter=args.resolution,
        n_iter_leiden=args.n_iter_leiden,
        random_seed=args.random_seed,
        num_threads=args.num_threads,
        distance="l2",
    )
    model.run_PARC()
    parc_elapsed = time.time() - t_parc
    labels_raw = np.asarray(model.labels)
    labels = renumber_by_size(labels_raw)
    print(f"[INFO] PARC elapsed={parc_elapsed:.1f}s clusters={len(np.unique(labels))}", flush=True)

    cluster_csv = outdir / f"{run_name}_clusters.csv"
    with open(cluster_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "cluster"])
        writer.writerows(zip(ids, labels.tolist()))
    print(f"[INFO] Wrote {cluster_csv}", flush=True)

    sil = "NA"
    if len(np.unique(labels)) > 1:
        rng = np.random.default_rng(args.random_seed)
        sample_n = min(args.silhouette_max, vis.shape[0])
        idx = np.sort(rng.choice(vis.shape[0], size=sample_n, replace=False))
        try:
            sil = float(silhouette_score(vis[idx], labels[idx], metric="euclidean"))
        except Exception as exc:
            print(f"[WARN] silhouette failed: {exc}", flush=True)
            sil = "NA"

    sizes = sorted([(int(cid), int(np.sum(labels == cid))) for cid in np.unique(labels)], key=lambda x: -x[1])
    png = outdir / f"{run_name}.png"
    title = f"PARC full vis | knn={args.knn}, resolution={args.resolution}, clusters={len(sizes)}"
    plot_clusters(vis, labels, png, title, args.plot_max, args.random_seed)
    print(f"[INFO] Wrote {png}", flush=True)

    summary_csv = outdir / f"{run_name}_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "solution", "method", "cells", "features", "knn", "resolution", "n_iter_leiden",
            "num_threads", "clusters", "silhouette_sample", "parc_elapsed_seconds",
            "cluster_sizes", "png", "csv",
        ])
        writer.writerow([
            run_name, "PARC", vis.shape[0], vis.shape[1], args.knn, args.resolution,
            args.n_iter_leiden, args.num_threads, len(sizes), sil, parc_elapsed,
            ";".join([f"{cid}:{count}" for cid, count in sizes]), str(png), str(cluster_csv),
        ])
    print(f"[INFO] Wrote {summary_csv}", flush=True)
    print("[INFO] Done", flush=True)


if __name__ == "__main__":
    main()
