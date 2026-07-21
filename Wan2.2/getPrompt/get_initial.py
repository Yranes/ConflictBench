import os
import json
import subprocess
import sys

# Should Run in Wan2.2
BASE_RESULT_DIR = "../TextWorld/TestWorld2/results"
SCRIPT_DIR = "getPrompt"

def get_env_json_path(ep_id):
    if ep_id.startswith("EP1"):
        return os.path.join(BASE_RESULT_DIR, "ulxEnv", ep_id, "env.json")
    elif ep_id.startswith("EP2"):
        return os.path.join(BASE_RESULT_DIR, "EP2-Result", "ulxEnv", ep_id, "env.json")
    elif ep_id.startswith("EP3"):
        return os.path.join(BASE_RESULT_DIR, "EP3-Result", "ulxEnv", ep_id, "env.json")
    else:
        raise ValueError(f"Unrecognized EP prefix: {ep_id}")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run_command(command):
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_initial.py <EPx-xxx>")
        print("Example: python get_initial.py EP1-005")
        sys.exit(1)

    ep_id = sys.argv[1]
    env_json_path = get_env_json_path(ep_id)

    print(f"Target EP: {ep_id}")
    print(f"Target JSON: {env_json_path}")

    if not os.path.exists(env_json_path):
        print(f"Error: config file not found {env_json_path}")
        sys.exit(1)

    data = load_json(env_json_path)
    t2v_prompt = ""

    if "t2v_description" in data:
        print("'t2v_description' already exists, skipping generation.")
        return
    else:
        print("'t2v_description' missing, generating...")
        run_command(["python", os.path.join(SCRIPT_DIR, "get_t2v_prompt.py"), ep_id])

        data = load_json(env_json_path)
        if "t2v_description" not in data:
            print("Error: script finished but 't2v_description' still missing.")
            sys.exit(1)
        t2v_prompt = data["t2v_description"]

    print(f"Prompt preview: {t2v_prompt[:50]}...")

    print("Generating I2V prompt...")
    run_command(["python", os.path.join(SCRIPT_DIR, "get_i2v_prompt.py"), ep_id])

    print("Running test_t2v.sh to generate video...")
    captured_lines = []

    try:
        process = subprocess.Popen(
            ["bash", "test_t2v.sh", t2v_prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                print(line, end='')
                stripped_line = line.strip()
                if stripped_line:
                    captured_lines.append(stripped_line)

        return_code = process.wait()

        if return_code != 0:
            print(f"\nError: test_t2v.sh failed with return code {return_code}")
            sys.exit(1)

        if not captured_lines:
            print("Error: test_t2v.sh produced no output, cannot extract video path.")
            sys.exit(1)

        video_path = captured_lines[-1]
        print(f"\nCaptured video path: {video_path}")

        print(f"Writing 'initial_video_path' to {env_json_path} ...")
        current_data = load_json(env_json_path)
        current_data["initial_video_path"] = video_path
        save_json(env_json_path, current_data)
        print("Update complete.")

    except Exception as e:
        print(f"Exception during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
