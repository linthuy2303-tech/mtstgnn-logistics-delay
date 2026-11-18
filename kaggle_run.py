#!/usr/bin/env python3
"""Job queue for a machine with several GPUs (written for the 2 x T4 of a Kaggle notebook).

It issues the same commands as run_plan.sh (same train.py arguments, same posthoc.py call) but keeps one training per GPU
running (CUDA_VISIBLE_DEVICES), orders the runs so that the GPUs finish together, and chooses the beta of the tuned MT-F
by itself from the validation losses, by the rule of aggregate.py (lowest mean best validation loss over the seeds that
every beta has). A run whose JSON exists is skipped, so the script can simply be started again after an interruption.

    python kaggle_run.py --stages grid,tuned,core,sens,e100 --out results --b1-val b1_val.json [--dry-run]

Stages (METR-LA, float16 encoder, 50 epochs unless stated):
  grid    MT-F with beta in {0.25, 0.5, 2, 4}, seeds 0-2                       -> OUT          12 runs
  tuned   MT-F with the beta chosen from grid + beta = 1, seeds 3-5            -> OUT           3 runs (none if beta = 1 wins)
  core    R, C, MT-F (beta = 1), MT-U, MT-P, seeds 3-5                         -> OUT          15 runs
  sens    label-rule variants and two ablations (run_plan.sh 'sens'), seed 0   -> OUT          10 runs
  e100    the five core variants on seed 0 with up to 100 epochs               -> OUT_e100      5 runs
  xcheck  R and MT-F (beta = 1) on seeds 0-2, a check of the platform          -> OUT_xcheck    6 runs
Seed-0 predictions (.npz) are kept for the grid only; the other stages delete them after posthoc.py has run.
The beta = 1 validation losses of seeds 0-2 come from the earlier Colab runs (--b1-val, a JSON {"0": v, "1": v, "2": v}).
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile

D, DIR = "metr-la", "data/METR-LA"
GRID = ["0.25", "0.5", "2", "4"]
EST_MIN = {"reg": 16.0, "cls": 16.5, "mtf": 17.5, "mtu": 17.5, "mtp": 26.0}   # minutes per 50-epoch run on a T4 (ordering only)
LOCK = threading.Lock()
EXPORT_LOCK = threading.Lock()


class Job:
    def __init__(self, stage, name, seed, args, out, epochs=50, keep_npz=False):
        self.stage, self.name, self.seed, self.args = stage, name, seed, args
        self.out, self.epochs, self.keep_npz = out, epochs, keep_npz
        self.base = f"{out}/{name}_s{seed}"
        self.key = f"{os.path.basename(out)}/{name}_s{seed}"

    def done(self):
        return os.path.exists(self.base + ".json")

    def minutes(self):
        return EST_MIN.get(self.name.split("_")[0], 20.0) * self.epochs / 50.0

    def command(self):
        return ["python", "train.py", "--data", D, "--data_dir", DIR, "--epochs", str(self.epochs), "--log_grad_cos",
                "--seed", str(self.seed), "--amp", "--save_preds", "--out", self.base + ".json"] + self.args


# ---- the runs of each stage (the same arguments as the stages of run_plan.sh)
def core_jobs(stage, seeds, out, epochs=50, keep_npz=False, only=None):
    jobs = []
    for s in seeds:
        for name, args in [("reg", ["--mode", "reg"]),
                           ("cls", ["--mode", "cls", "--class_weight"]),
                           ("mtf_b1", ["--mode", "multi", "--alpha", "1", "--beta", "1", "--class_weight"]),
                           ("mtu", ["--mode", "multi", "--weighting", "uncertainty", "--class_weight"]),
                           ("mtp", ["--mode", "multi", "--pcgrad", "--class_weight"])]:
            if only is None or name in only:
                jobs.append(Job(stage, name, s, args, out, epochs, keep_npz))
    return jobs


def grid_jobs(stage, betas, seeds, out, keep_npz=False):
    return [Job(stage, f"mtf_b{b}", s, ["--mode", "multi", "--alpha", "1", "--beta", b, "--class_weight"], out,
                keep_npz=keep_npz and s == 0) for b in betas for s in seeds]


def sens_jobs(stage, seeds, out):
    jobs = []
    for s in seeds:
        for tag, extra in [("k1", ["--persist_k", "1"]), ("k6", ["--persist_k", "6"]),
                           ("t60_30", ["--t_warn", "0.60", "--t_crit", "0.30"]),
                           ("t80_50", ["--t_warn", "0.80", "--t_crit", "0.50"])]:
            jobs.append(Job(stage, f"cls_{tag}", s, ["--mode", "cls", "--class_weight"] + extra, out))
            jobs.append(Job(stage, f"mtp_{tag}", s, ["--mode", "multi", "--pcgrad", "--class_weight"] + extra, out))
        jobs.append(Job(stage, "mtp_noadj", s, ["--mode", "multi", "--pcgrad", "--class_weight", "--no_adaptive"], out))
        jobs.append(Job(stage, "mtp_nocw", s, ["--mode", "multi", "--pcgrad"], out))
    return jobs


# ---- choice of the tuned beta (the rule of aggregate.py)
def best_val(path):
    return min(h["val"] for h in json.load(open(path))["history"])


def choose_beta(out, b1_val):
    runs = {}
    for p in glob.glob(f"{out}/mtf_b*_s*.json"):
        m = re.match(r".*/mtf_b([0-9.]+)_s(\d+)\.json$", p)
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = best_val(p)
    runs["1"] = {int(k): float(v) for k, v in b1_val.items()}          # beta = 1 on seeds 0-2: earlier Colab runs
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if not common:
        return "1", {}, []
    score = {b: sum(v[s] for s in common) / len(common) for b, v in runs.items()}
    return min(score, key=score.get), score, common


# ---- bookkeeping
def log(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class State:
    def __init__(self, path):
        self.path, self.d = path, dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), stage=None, running={}, finished={}, failed=[], decisions={})
        if os.path.exists(path):                                                      # restarted after an interruption: keep what is known
            try:
                old = json.load(open(path))
                self.d.update(finished=old.get("finished", {}), decisions=old.get("decisions", {}), first_started=old.get("first_started", old.get("started")))
            except Exception:
                pass

    def save(self):
        with LOCK:
            tmp = self.path + ".tmp"
            json.dump(self.d, open(tmp, "w"), indent=1)
            os.replace(tmp, self.path)


def thin_zip(src_dir, dst, extra=()):
    n, tmp = 0, dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(src_dir + "/*.json")):
            try:
                r = json.load(open(p))
                if "grad_cos" in r:
                    r["grad_cos"] = [e for e in r["grad_cos"] if e[1] % 20 == 0]      # every 20th step, as in the Colab export
                z.writestr(os.path.basename(p), json.dumps(r)); n += 1
            except Exception:
                pass                                                                  # a file that is being written right now
        for p in extra:
            if os.path.exists(p):
                z.write(p, os.path.basename(p))
    os.replace(tmp, dst)
    return n


def export_all(args, state, quiet=False):
    extra = [args.status, args.env, args.ds]
    with EXPORT_LOCK:
        try:
            n = thin_zip(args.out, f"{args.export}/results_thin.zip", extra)
            for tag, d in (("e100", args.out_e100), ("xcheck", args.out_xcheck)):
                if os.path.isdir(d) and glob.glob(d + "/*.json"):
                    thin_zip(d, f"{args.export}/results_{tag}_thin.zip")
            if not quiet:
                log(f"export: {n} run files")
        except Exception as e:                                                        # never stop the queue because of the export
            log(f"export failed: {e!r}")


def run_job(job, gpu, args, state):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    os.makedirs(job.out + "/logs", exist_ok=True)
    t0 = time.time()
    with LOCK:
        state.d["running"][job.key] = dict(gpu=gpu, start=time.strftime("%H:%M:%S"))
    state.save()
    log(f"start  {job.key} on GPU {gpu}")
    ok = False
    try:
        with open(f"{job.out}/logs/{job.name}_s{job.seed}.log", "w") as lf:
            r = subprocess.run(job.command(), env=env, stdout=lf, stderr=subprocess.STDOUT)
            ok = r.returncode == 0 and job.done()
            if ok and os.path.exists(job.base + ".npz"):
                subprocess.run(["python", "posthoc.py", "--npz", job.base + ".npz", "--out", job.base + "_post.json"],
                               env=env, stdout=lf, stderr=subprocess.STDOUT)
                if not job.keep_npz:
                    os.remove(job.base + ".npz")
    except Exception as e:
        log(f"ERROR {job.key}: {e!r}")
    mins = (time.time() - t0) / 60
    with LOCK:
        state.d["running"].pop(job.key, None)
        if ok:
            state.d["finished"][job.key] = dict(gpu=gpu, minutes=round(mins, 1), end=time.strftime("%H:%M:%S"))
            if job.key in state.d["failed"]:
                state.d["failed"].remove(job.key)
        elif job.key not in state.d["failed"]:
            state.d["failed"].append(job.key)
    state.save()
    log(f"{'done  ' if ok else 'FAILED'} {job.key} on GPU {gpu} ({mins:.1f} min)")
    export_all(args, state, quiet=True)
    return ok


def run_stage(name, jobs, args, state, ngpu):
    todo = [j for j in jobs if not j.done()]
    todo.sort(key=lambda j: -j.minutes())                                             # longest first: the GPUs end together
    state.d["stage"] = name
    state.save()
    log(f"=== stage {name}: {len(jobs)} runs, {len(jobs) - len(todo)} already finished")
    if args.dry_run:
        for j in todo:
            print(" ".join(j.command()))
        return
    for attempt in (1, 2):                                                            # one retry of runs that failed
        if not todo:
            break
        free = queue.Queue()
        for g in range(ngpu):
            free.put(g)

        def work(j):
            g = free.get()
            try:
                return run_job(j, g, args, state)
            finally:
                free.put(g)

        with cf.ThreadPoolExecutor(max_workers=ngpu) as ex:
            list(ex.map(work, todo))
        export_all(args, state)
        todo = [j for j in todo if not j.done()]
        if todo:
            log(f"{len(todo)} run(s) of stage {name} did not finish" + (" - trying once more" if attempt == 1 else ""))
    log(f"=== stage {name} finished")


def n_gpus():
    r = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
    return max(1, len([l for l in r.stdout.splitlines() if l.startswith("GPU ")]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="grid,tuned,core,sens,e100")
    ap.add_argument("--gpus", type=int, default=0, help="0 = all GPUs that nvidia-smi lists")
    ap.add_argument("--out", default="results")
    ap.add_argument("--out-e100", default=None)
    ap.add_argument("--out-xcheck", default=None)
    ap.add_argument("--export", default="export")
    ap.add_argument("--b1-val", default="b1_val.json")
    ap.add_argument("--env", default="export/env_kaggle.json")
    ap.add_argument("--ds", default="export/ds_metr_kaggle.json")
    ap.add_argument("--status", default="export/status_kaggle.json")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per training process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out_e100 = args.out_e100 or args.out.rstrip("/") + "_e100"
    args.out_xcheck = args.out_xcheck or args.out.rstrip("/") + "_xcheck"
    os.makedirs(args.export, exist_ok=True)
    ngpu = args.gpus or n_gpus()
    state = State(args.status)
    log(f"{ngpu} GPU(s); stages {args.stages}")
    b1_val = json.load(open(args.b1_val))
    for st in args.stages.split(","):
        if st == "grid":
            run_stage("grid", grid_jobs("grid", GRID, [0, 1, 2], args.out, keep_npz=True), args, state, ngpu)
            best, score, common = choose_beta(args.out, b1_val)
            state.d["decisions"]["tuned_beta"] = dict(best=best, val_loss_by_beta=score, seeds=common)
            state.save()
            log(f"validation loss by beta (seeds {common}): { {b: round(v, 4) for b, v in score.items()} } -> tuned beta {best}")
        elif st == "tuned":
            if "tuned_beta" not in state.d["decisions"]:
                best, score, common = choose_beta(args.out, b1_val)
                state.d["decisions"]["tuned_beta"] = dict(best=best, val_loss_by_beta=score, seeds=common)
                state.save()
            best = state.d["decisions"]["tuned_beta"]["best"]
            if best == "1":
                log("stage tuned: beta = 1 has the lowest validation loss, so the MT-F runs of the core stage are the tuned MT-F")
            else:
                run_stage("tuned", grid_jobs("tuned", [best], [3, 4, 5], args.out), args, state, ngpu)
        elif st == "core":
            run_stage("core", core_jobs("core", [3, 4, 5], args.out), args, state, ngpu)
        elif st == "sens":
            run_stage("sens", sens_jobs("sens", [0], args.out), args, state, ngpu)
        elif st == "e100":
            run_stage("e100", core_jobs("e100", [0], args.out_e100, epochs=100), args, state, ngpu)
        elif st == "xcheck":
            run_stage("xcheck", core_jobs("xcheck", [0, 1, 2], args.out_xcheck, only=("reg", "mtf_b1")), args, state, ngpu)
        else:
            sys.exit(f"unknown stage {st}")
        if not args.dry_run:
            export_all(args, state)
    state.d["stage"] = "ALL DONE"
    state.save()
    log("ALL STAGES DONE")


if __name__ == "__main__":
    main()
