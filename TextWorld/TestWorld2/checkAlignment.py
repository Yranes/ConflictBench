import json
import time
import os
import concurrent.futures
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

SYSTEM_PROMPT_VER2 = read_txt("./TextWorld/TestWorld2/prompt/AlignmentChecker_ver2.txt")

import re

def strip_code_blocks(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

def write_back_alignment_result(json_path, alignment_result):
    """
    Write the alignment result back to the original <model>.json file.
    Adds the following field to the original JSON:
        "alignment_check": {alignment_success, reasoning}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["alignment_check_think_only"] = alignment_result

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def evaluate_alignment(record, model="gpt-5.2-ca", max_retries=2):
    """
    record: dict containing scene/history
    """

    # print(json.dumps(record, ensure_ascii=False, indent=2))
    user_input = SYSTEM_PROMPT_VER2.format(history=json.dumps(record, ensure_ascii=False, indent=2))
    x = {"scene": record["scene"], "history": []}

    if isinstance(record["history"][0], dict) and "role" in record["history"][0]:
        for i in record["history"]:
            if i["role"] == "user":
                x["history"].append(i["content"])
            else:
                x["history"].append(i["think"])
                x["history"].append(i["action"])
        user_input = SYSTEM_PROMPT_VER2.format(history=json.dumps(x, ensure_ascii=False, indent=2))
    # print(user_input)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                temperature=0.0,
                reasoning_effort="low"
            )
            print(1)
            output = strip_code_blocks(response.choices[0].message.content)
            return json.loads(output)
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(0.5)
    return {"error": "failed after retries", "input": record}

def batch_evaluate(items, model="gpt-5.2-ca", max_workers=20):
    """
    items: list[(record, json_path)]
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(evaluate_alignment, {"scene": record.get("scene", record.get("scene_description")), "history": record["history"]}, model): (record, json_path)
            for record, json_path in items
        }

        for future in concurrent.futures.as_completed(future_to_item):
            record, json_path = future_to_item[future]
            try:
                alignment_result = future.result()
            except Exception as e:
                print("ERR")
                alignment_result = {"error": str(e)}

            write_back_alignment_result(json_path, alignment_result)

            print(f"[WRITE] result written back -> {json_path}")

def batch_evaluateMul(items, model="gpt-5", max_worker=5):
    pass


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

BASE = "./TextWorld/TestWorld2/results"

def load_episode_json(ep_id: str, model_name="deepseek-v3"):
    if not ep_id.startswith("EP"):
        raise ValueError(f"Invalid EP id format: {ep_id}")

    ep_num = ep_id.split("-")[0]  # EP1, EP2, EP3

    if ep_num == "EP1":
        search_path = f"{BASE}/result_history/{ep_id}/{model_name}.json"
    else:
        search_path = f"{BASE}/{ep_num}-Result/result_history/{ep_id}/{model_name}.json"

    if not os.path.isfile(search_path):
        print(f"[WARN] File not found: {search_path}")
        return None

    with open(search_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if data.get("good_end") is True:
                return None
            return data, search_path
        except Exception as e:
            print(f"[ERROR] JSON parse error in {search_path}: {e}")
            return None

def load_multiple_episodes(ep_prefix: str, st: int, ed: int, model_name="deepseek-v3"):
    items = []
    for i in range(st, ed + 1):
        ep_id = f"{ep_prefix}-{i:03d}"
        res = load_episode_json(ep_id, model_name=model_name)
        if res:
            data, json_path = res
            items.append((data, json_path))
    return items

def main(ep, model_name, st = 1, ed = 80):
    outputs = batch_evaluate(load_multiple_episodes(ep, st, ed, model_name))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
