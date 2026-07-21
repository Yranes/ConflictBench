import os
from openai import OpenAI
import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        str_s = json.load(f)
    return str_s

def read_prompt_txt(path): 
    with open(path, 'r') as f:
        out = f.read()
    return out

def write_json(path, s):
    str_s = json.dumps(s, ensure_ascii = False, indent = 4)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str_s)

# should run in Wan2.2
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

def get_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": prompt}
            ],
            reasoning_effort = "low"
        )
        resp = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        if "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        resp_json = json.loads(resp)
        return resp_json
    except Exception as e:
        print("Error during API call:", e, response)
        raise e
    return resp_json

def main(ep_id):
    if "EP2" in ep_id:
        base_path = "../TextWorld/TestWorld2/results/EP2-Result"
    elif "EP3" in ep_id:
        base_path = "../TextWorld/TestWorld2/results/EP3-Result"
    else:
        base_path = "../TextWorld/TestWorld2/results/"
    env_set = read_json(base_path + f"/ulxEnv/{ep_id}/env.json")
    opening = env_set.get("name", "Scene") + ": " + env_set["opening"]
    state = env_set["state"]
    action = read_json(base_path + f"/result_history/{ep_id}/deepseek-v3.json")["history"][0]
    t2v = env_set["t2v_description"]
    prompt = read_prompt_txt("./Wan2.2/getPrompt/template.md").format(SCENARIO=t2v, ACTIONS=action)
    if "EP3" in ep_id:
        prompt = read_prompt_txt("./Wan2.2/getPrompt/template_EP3.md").format(SCENARIO=t2v, ACTIONS=action)
    print("Prompt:", prompt)
    resp_json = get_response(prompt)
    if env_set.get("template"):
        return RuntimeError("i2v template is already exsist.")
    env_set["template"] = resp_json
    write_json(base_path + f"/ulxEnv/{ep_id}/env.json", env_set)
    print(base_path + f"/ulxEnv/{ep_id}/env.json")

import fire

if __name__ == "__main__":
    fire.Fire(main)