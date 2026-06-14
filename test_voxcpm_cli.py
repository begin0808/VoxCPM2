import os
import sys
import time

def run_test():
    print("=== Studio0808_VoxCPM 獨立推理測試 ===")
    
    # 1. 檢查 PyTorch & CUDA 狀態
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
    except ImportError:
        print("錯誤: 未能匯入 PyTorch。請檢查環境安裝。")
        sys.exit(1)

    # 2. 檢查 voxcpm
    try:
        from voxcpm import VoxCPM
        import soundfile as sf
        print("成功匯入 voxcpm 庫！")
    except ImportError:
        print("錯誤: 未能匯入 voxcpm 庫。")
        sys.exit(1)

    # 3. 尋找模型路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(script_dir, "models", "VoxCPM2")
    
    if os.path.exists(os.path.join(local_model_dir, "config.json")):
        model_source = local_model_dir
        print(f"載入本機模型: {model_source}")
    else:
        model_source = "openbmb/VoxCPM2"
        print(f"本機未檢測到模型，將自動從網路上載入: {model_source}")

    # 4. 初始化模型
    print("正在載入模型，請稍候（首次下載或載入可能需要數分鐘）...")
    start_time = time.time()
    try:
        model = VoxCPM.from_pretrained(model_source, load_denoiser=False)
        print(f"模型載入成功！耗時: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        sys.exit(1)

    # 5. 進行語音合成測試
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_output.wav")
    
    test_text = "(A young woman, gentle voice) 這是 Studio0808 VoxCPM 語音合成的第一次推論測試。"
    print(f"測試文字: {test_text}")
    print("正在進行語音合成...")
    
    infer_start = time.time()
    try:
        wav = model.generate(
            text=test_text,
            cfg_value=2.0,
            inference_timesteps=10,
        )
        sf.write(output_file, wav, model.tts_model.sample_rate)
        print(f"語音合成成功！")
        print(f"輸出檔案路徑: {output_file}")
        print(f"推論耗時: {time.time() - infer_start:.2f} 秒")
        print("=== 測試完成 ===")
    except Exception as e:
        print(f"語音合成推論失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
