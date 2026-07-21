import os
import subprocess
import sys
import argparse
import time

# Should Run in Wan2.2
TARGET_SCRIPT_PATH = "./getPrompt/get_initial.py"

# EP ID 的前缀
EP_PREFIX = "EP2" 
PADDING_ZEROS = 3 

def run_command(command):
    """运行命令并实时打印输出"""
    print(f"\n[Batch] 正在执行: {' '.join(command)}")
    try:
        # check=True 会在脚本返回非0状态码时抛出异常
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Batch] ❌ 执行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[Batch] 用户中断执行。")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="批量运行 get_initial.py 脚本")
    parser.add_argument("start", type=int, help="开始序号 (例如 1)")
    parser.add_argument("end", type=int, help="结束序号 (例如 5)")
    parser.add_argument("--prefix", type=str, default=EP_PREFIX, help=f"EP前缀 (默认: {EP_PREFIX})")
    
    args = parser.parse_args()

    if not os.path.exists(TARGET_SCRIPT_PATH):
        print(f"错误: 找不到目标脚本文件: {TARGET_SCRIPT_PATH}")
        sys.exit(1)

    success_count = 0
    fail_count = 0
    failed_eps = []

    print(f"=== 开始批量任务: {args.prefix} | 范围: {args.start} -> {args.end} ===\n")

    # 遍历范围 (包含 end)
    for i in range(args.start, args.end + 1):
        ep_num = str(i).zfill(PADDING_ZEROS)
        ep_id = f"{args.prefix}-{ep_num}"
        
        cmd = ["python", TARGET_SCRIPT_PATH, ep_id]
        
        # 执行
        if run_command(cmd):
            print(f"[Batch] ✅ {ep_id} 完成。")
            success_count += 1
        else:
            print(f"[Batch] ❌ {ep_id} 失败。")
            fail_count += 1
            failed_eps.append(ep_id)

    print("\n" + "="*40)
    print(f"批量任务结束。")
    print(f"总计: {args.end - args.start + 1}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    if failed_eps:
        print(f"失败的 EP 列表: {failed_eps}")
    print("="*40)

if __name__ == "__main__":
    main()