#!/usr/bin/env python3
import subprocess
import os
import time
from tqdm import tqdm
import argparse

# --- Configuration ---

# Default SEED_ID lists grouped by episode
SEED_IDS_EP1 = [f"EP1-{i:03d}" for i in range(1, 80)]
SEED_IDS_EP2 = [f"EP2-{i:03d}" for i in range(1, 63)]
SEED_IDS_EP3 = [f"EP3-{i:03d}" for i in range(1, 60)]
SEED_IDS = SEED_IDS_EP1 + SEED_IDS_EP2 + SEED_IDS_EP3

# RESUME_IDS = ["EP1-028", "EP1-060", ...]

EVAL_MULAGENT_PATH = "./TextWorld/TestWorld2/bash/evalMulAgent.sh"

RESULT_HISTORY_DIR = "./TextWorld/TestWorld2/results/result_history"
RESULT_HISTORY_DIR_EP2 = "./TextWorld/TestWorld2/results/EP2-Result/result_history"
RESULT_HISTORY_DIR_EP3 = "./TextWorld/TestWorld2/results/EP3-Result/result_history"


def check_is_evaluated(seed_id, model_name):
    """
    Check whether the given SEED_ID and model_name have already been evaluated.
    Judged by whether the corresponding result file exists in RESULT_HISTORY_DIR.
    """
    result_file = os.path.join(RESULT_HISTORY_DIR, f"{seed_id}", f"{model_name}.json")
    if "EP2" in seed_id:
        result_file = os.path.join(RESULT_HISTORY_DIR_EP2, f"{seed_id}", f"{model_name}.json")
    elif "EP3" in seed_id:
        result_file = os.path.join(RESULT_HISTORY_DIR_EP3, f"{seed_id}", f"{model_name}.json")
    return os.path.isfile(result_file)


def check_is_legal(seed_id):
    """Check whether a compiled .ulx environment exists for the given seed_id."""
    ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/ulxEnv", seed_id)
    if "EP2" in seed_id:
        ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/EP2-Result/ulxEnv", seed_id)
    elif "EP3" in seed_id:
        ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/EP3-Result/ulxEnv", seed_id)
    return os.path.isdir(ulx_file_dir)


def run_bash_script(script_path, seed_id, model_name, server_url, resume_flag) -> tuple:
    """
    Run the bash script for a single SEED_ID and return the result.
    Streams subprocess output in real time.
    """
    command = ["/bin/bash", script_path, seed_id, model_name, server_url, resume_flag]
    start_time = time.time()

    # 1. Copy current environment variables.
    my_env = os.environ.copy()
    # 2. Force Python unbuffered mode so that output from the bash script's
    #    child python processes is flushed immediately.
    my_env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,        # Read the pipe line by line
            encoding='utf-8',
            errors='replace',
            env=my_env        # Inject the environment variables
        )

        captured_output = []

        # Read and forward output in real time
        for line in process.stdout:
            print(line, end='', flush=True)
            captured_output.append(line)

        process.wait()

        elapsed = time.time() - start_time
        full_log = "".join(captured_output)

        if process.returncode == 0:
            return ("success", seed_id, full_log, "", elapsed)
        else:
            return ("error", seed_id, full_log, f"Exit code: {process.returncode}", elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        return ("error", seed_id, "", str(e), elapsed)


def compute_success_rate(seed_ids: list, model_name: str) -> tuple:
    """Compute the success rate for the given SEED_IDS and model_name."""
    import json
    cnt = 0
    success_cnt = 0
    for seed_id in seed_ids:
        result_file = os.path.join(RESULT_HISTORY_DIR, f"{seed_id}", f"{model_name}.json")
        if not os.path.isfile(result_file):
            print(f"    [WARN] File not found: {seed_id} @ {model_name}")
            continue
        cnt += 1
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("good_end"):
                success_cnt += 1
            else:
                print(f"    [FAIL] {model_name} failed case: {seed_id} @ {model_name}")
    return success_cnt, cnt, (success_cnt / cnt if cnt > 0 else 0.0)


def main():
    argparser = argparse.ArgumentParser(
        description="Run multimodal agent evaluation sequentially.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    argparser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Multimodal model name to evaluate'
    )
    argparser.add_argument(
        '-s', '--server',
        type=str,
        required=True,
        help='Full inference server URL (e.g. http://gpu09:8021/generate)'
    )
    argparser.add_argument(
        '-e', '--episode',
        type=str,
        nargs='+',
        choices=['EP1', 'EP2', 'EP3'],
        default=None,
        help='Episode groups to evaluate (e.g., -e EP1, -e EP1 EP3). '
             'If not provided, all EP1/EP2/EP3 are tested.'
    )
    argparser.add_argument(
        '-f', '--force',
        action='store_true',  # When -f is provided, args.force will be True
        default=False,
        help='Force re-evaluation even if results exist (default: False)'
    )
    argparser.add_argument(
        '-r', '--resume',
        action='store_true',
        default=False,
        help='Resume from previous incomplete runs; implies --force (default: False)'
    )
    argparser.add_argument(
        'seed_ids',
        nargs='*',  # zero or more
        default=SEED_IDS,
        help='(Optional) List of SEED_IDs to process. Cannot be used together with -e/--episode.'
    )
    args = argparser.parse_args()

    if args.seed_ids != SEED_IDS and args.episode:
        argparser.error("-e/--episode and positional seed_ids are mutually exclusive.")

    server_url = args.server
    script_to_run = EVAL_MULAGENT_PATH
    if not os.path.exists(script_to_run):
        print(f"[ERROR] Evaluation script not found: {script_to_run}")
        return

    MODEL_NAME = args.model
    FORCE_RUN = args.force
    RESUME_RUN = args.resume

    if RESUME_RUN:
        FORCE_RUN = True
        print("[WARN] --resume is enabled, --force is implied.")

    if args.episode:
        ep_map = {'EP1': SEED_IDS_EP1, 'EP2': SEED_IDS_EP2, 'EP3': SEED_IDS_EP3}
        all_seed_ids = []
        for ep in args.episode:
            all_seed_ids += ep_map[ep]
    else:
        all_seed_ids = args.seed_ids

    print(f"[INFO] Start evaluating model: '{MODEL_NAME}'")
    print(f"   Server URL: {server_url}")
    print(f"   Force re-evaluation: {'yes' if FORCE_RUN else 'no'}")
    print(f"   Resume: {'yes' if RESUME_RUN else 'no'}")
    if args.episode:
        print(f"   Scope: {' '.join(args.episode)}")
    else:
        print(f"   Scope: EP1 EP2 EP3 (all)")

    tasks_to_run = []
    tasks_skipped = []

    for seed_id in all_seed_ids:
        if not check_is_legal(seed_id):
            print(f"   [SKIP] Not compiled / invalid env, skipping: {seed_id}")
            tasks_skipped.append(seed_id)
    all_seed_ids = [sid for sid in all_seed_ids if sid not in tasks_skipped]

    if FORCE_RUN:
        print(f"   -f/--force enabled, will run all {len(all_seed_ids)} tasks.")
        tasks_to_run = all_seed_ids
    else:
        print(f"   Checking {len(all_seed_ids)} tasks for existing results...")
        for seed_id in all_seed_ids:
            if check_is_evaluated(seed_id, MODEL_NAME):
                tasks_skipped.append(seed_id)
            else:
                tasks_to_run.append(seed_id)
        print(f"   {len(tasks_to_run)} new tasks to run. {len(tasks_skipped)} existing tasks will be skipped.")

    if not tasks_to_run:
        print("\n[INFO] No new tasks to run. Exiting.")
        return

    success_count = 0
    error_count = 0
    resume_flag = "--resume=True" if RESUME_RUN else "--resume=False"

    for seed_id in tqdm(tasks_to_run, desc=f"Evaluating {MODEL_NAME}"):
        try:
            print("now eval: " + seed_id)
            status, returned_seed_id, stdout, stderr, elapsed = run_bash_script(
                script_to_run, seed_id, MODEL_NAME, server_url, resume_flag
            )
        except Exception as e:
            # Catch per-task exceptions so a single failure does not abort the whole loop.
            status = "critical_error"
            returned_seed_id = seed_id
            stdout = ""
            stderr = str(e)
            elapsed = 0.0

        if status == "success":
            success_count += 1
            print("\n" + "------------------------------------------------------")
            print(f"[OK] Success: {returned_seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
            print("--- STDOUT: ---")
            print(stdout.strip())
            if stderr:
                print("--- STDERR (Warning): ---")
                print(stderr.strip())
            print("------------------------------------------------------")

        elif status == "error":
            error_count += 1
            print("\n" + "======================================================")
            print(f"[FAIL] {returned_seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
            print("--- STDOUT: ---")
            print(stdout.strip())
            print("--- STDERR (Error): ---")
            print(stderr.strip())
            print("======================================================")

        elif status == "critical_error":
            error_count += 1
            print("\n" + "======================================================")
            print(f"[CRITICAL] {returned_seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
            print(f"--- Python Exception: {stderr.strip()} ---")
            print("======================================================")

    print(f"\n[INFO] Evaluation of '{MODEL_NAME}' finished.")
    print(f"   Total: {success_count} succeeded, {error_count} failed (out of {len(tasks_to_run)} tasks).")
    if not FORCE_RUN:
        print(f"   Skipped: {len(tasks_skipped)} tasks with existing results.")


if __name__ == "__main__":
    main()
