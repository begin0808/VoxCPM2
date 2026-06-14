import os
import sys

# Set HF_ENDPOINT to hf-mirror for extremely fast downloads in Asia
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_voxcpm(use_mirror=True, use_modelscope=True, local_dir="./models/VoxCPM2"):
    """
    Downloads the VoxCPM2 model weights.
    """
    print(f"=== VoxCPM2 模型下載器 ===")
    print(f"目標下載路徑: {os.path.abspath(local_dir)}")
    
    os.makedirs(local_dir, exist_ok=True)

    if use_mirror:
        print("正在嘗試自 Hugging Face 鏡像站 (hf-mirror.com) 下載模型 (推薦，頻寬較大)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="openbmb/VoxCPM2", local_dir=local_dir)
            print("鏡像站下載成功！")
            return True
        except Exception as e:
            print(f"鏡像站下載失敗: {e}")
            print("將切換為 ModelScope 下載...")

    if use_modelscope:
        print("正在嘗試自 ModelScope 下載模型...")
        try:
            from modelscope import snapshot_download
            snapshot_download("OpenBMB/VoxCPM2", local_dir=local_dir)
            print("ModelScope 下載成功！")
            return True
        except Exception as e:
            print(f"ModelScope 下載失敗: {e}")
            print("將切換為官方 Hugging Face 下載...")

    print("正在嘗試自 Hugging Face 下載模型...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="openbmb/VoxCPM2", local_dir=local_dir)
        print("Hugging Face 下載成功！")
        return True
    except Exception as e:
        print(f"Hugging Face 下載失敗: {e}")
        return False

if __name__ == "__main__":
    # 決定專案根目錄
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(script_dir, "models", "VoxCPM2")
    
    success = download_voxcpm(use_modelscope=True, local_dir=target_dir)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
