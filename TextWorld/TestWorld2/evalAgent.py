#!/usr/bin/env python3
import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# --- Configuration ---

# Default SEED_ID lists grouped by episode
SEED_IDS_EP1 = [f"EP1-{i:03d}" for i in range(1, 80)]
SEED_IDS_EP2 = [f"EP2-{i:03d}" for i in range(1, 62)]
SEED_IDS_EP3 = [f"EP3-{i:03d}" for i in range(1, 60)]
SEED_IDS = SEED_IDS_EP1 + SEED_IDS_EP2 + SEED_IDS_EP3

PIPLINE_PATH = "./TextWorld/TestWorld2/bash/generateAll.sh"
LOOP_COMPILE_PATH = "./TextWorld/TestWorld2/bash/loopCompile.sh"
EVAL_AGENT_PATH = "./TextWorld/TestWorld2/bash/evalAgent.sh"

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
    ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/ulxEnv", seed_id)
    if "EP2" in seed_id:
        ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/EP2-Result/ulxEnv", seed_id)
    elif "EP3" in seed_id:
        ulx_file_dir = os.path.join("./TextWorld/TestWorld2/results/EP3-Result/ulxEnv", seed_id)
    if not os.path.isdir(ulx_file_dir):
        return False
    return True

def run_bash_script(script_path, seed_id, model_name):
    """
    Run the bash script for a single SEED_ID and return the result.
    This function runs in a separate thread.

    Note: the bash script receives seed_id as $1 and model_name as $2.
    """
    # Pass model_name as the second argument to the bash script
    command = ["/bin/bash", script_path, seed_id, model_name]
    start_time = time.time()

    try:
        # Run the subprocess.
        # check=True raises CalledProcessError when the script exits with a non-zero code.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        elapsed = time.time() - start_time
        return ("success", seed_id, result.stdout, result.stderr, elapsed)

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        return ("error", seed_id, e.stdout, e.stderr, elapsed)

    except Exception as e:
        # Catch other unexpected errors (e.g., script file not found).
        elapsed = time.time() - start_time
        return ("critical_error", seed_id, "", str(e), elapsed)

def compute_success_rate(seed_ids: list, model_name: str) -> tuple:
    """
    Compute the success rate for the given SEED_IDS and model_name.
    """
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
        description="Run agent evaluation in parallel.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    argparser.add_argument(
        '-w', '--workers', 
        type=int, 
        default=5, 
        help='Number of concurrent threads (default: 5)'
    )
    argparser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Model name to evaluate (e.g., gpt-4, gpt-3.5-turbo)'
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
        'seed_ids',
        nargs='*',  # zero or more
        default=SEED_IDS,
        help='(Optional) List of SEED_IDs to process. Cannot be used together with -e/--episode.'
    )
    args = argparser.parse_args()

    if args.seed_ids != SEED_IDS and args.episode:
        argparser.error("-e/--episode and positional seed_ids are mutually exclusive.")

    script_to_run = EVAL_AGENT_PATH
    if not os.path.exists(script_to_run):
        print(f"[ERROR] Evaluation script not found: {script_to_run}")
        return

    MAX_THREADS = args.workers
    MODEL_NAME = args.model
    FORCE_RUN = args.force

    if args.episode:
        ep_map = {'EP1': SEED_IDS_EP1, 'EP2': SEED_IDS_EP2, 'EP3': SEED_IDS_EP3}
        all_seed_ids = []
        for ep in args.episode:
            all_seed_ids += ep_map[ep]
    else:
        all_seed_ids = args.seed_ids

    print(f"[INFO] Start evaluating model: '{MODEL_NAME}'")
    print(f"   Threads: {MAX_THREADS}")
    print(f"   Force re-evaluation: {'yes' if FORCE_RUN else 'no'}")
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

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:

        tasks = {
            executor.submit(run_bash_script, script_to_run, seed_id, MODEL_NAME): seed_id
            for seed_id in tasks_to_run
        }

        for future in tqdm(as_completed(tasks), total=len(tasks), desc=f"Evaluating {MODEL_NAME}"):

            status, seed_id, stdout, stderr, elapsed = future.result()

            if status == "success":
                success_count += 1
                print("\n" + "------------------------------------------------------")
                print(f"[OK] Success: {seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
                print("--- STDOUT: ---")
                print(stdout.strip())
                if stderr:
                    print("--- STDERR (Warning): ---")
                    print(stderr.strip())
                print("------------------------------------------------------")

            elif status == "error":
                error_count += 1
                print("\n" + "======================================================")
                print(f"[FAIL] {seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
                print("--- STDOUT: ---")
                print(stdout.strip())
                print("--- STDERR (Error): ---")
                print(stderr.strip())
                print("======================================================")

            elif status == "critical_error":
                error_count += 1
                print("\n" + "======================================================")
                print(f"[CRITICAL] {seed_id} @ {MODEL_NAME} (elapsed: {elapsed:.2f}s)")
                print(f"--- Python Exception: {stderr.strip()} ---")
                print("======================================================")

    print(f"\n[INFO] Evaluation of '{MODEL_NAME}' finished.")
    print(f"   Total: {success_count} succeeded, {error_count} failed (out of {len(tasks_to_run)} tasks).")
    if not FORCE_RUN:
        print(f"   Skipped: {len(tasks_skipped)} tasks with existing results.")

if __name__ == "__main__":
    main()