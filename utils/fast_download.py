import os
import sys
import time
import urllib.request

def download_file(url, filepath):
    print(f"正在下載: {url} -> {filepath}")
    start_time = time.time()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if target already exists and has size
    if os.path.exists(filepath):
        # We can resume or overwrite. For simplicity, we overwrite unless it's fully downloaded.
        # But since we want to be safe, if it exists, we overwrite.
        pass

    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            bytes_so_far = 0
            block_size = 1024 * 1024 # 1MB chunks
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_so_far += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed = (bytes_so_far / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    
                    if total_size > 0:
                        percent = float(bytes_so_far) / total_size * 100
                        sys.stdout.write(f"\r下載中: {bytes_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.2f}%) | 速度: {speed:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\r下載中: {bytes_so_far / (1024*1024):.1f}MB | 速度: {speed:.2f} MB/s")
                    sys.stdout.flush()
        print(f"\n下載完成！耗時: {time.time() - start_time:.2f} 秒\n")
        return True
    except Exception as e:
        print(f"\n下載失敗: {e}\n")
        return False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(script_dir, "models", "VoxCPM2")
    
    files_to_download = {
        "audiovae.pth": "https://hf-mirror.com/openbmb/VoxCPM2/resolve/main/audiovae.pth",
        "model.safetensors": "https://hf-mirror.com/openbmb/VoxCPM2/resolve/main/model.safetensors"
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(target_dir, filename)
        # Check if file already downloaded (checking size)
        # audiovae.pth size is ~359MB (377030000 bytes approx)
        # model.safetensors size is ~4.27GB (4580000000 bytes approx)
        expected_min_size = 300 * 1024 * 1024 if filename == "audiovae.pth" else 4 * 1024 * 1024 * 1024
        
        if os.path.exists(filepath) and os.path.getsize(filepath) >= expected_min_size:
            print(f"{filename} 已經存在且大小正確，跳過下載。")
            continue
            
        success = download_file(url, filepath)
        if not success:
            print(f"{filename} 下載失敗，退出。")
            sys.exit(1)
            
    print("所有權重檔案下載完成！")
    sys.exit(0)
