@echo off
chcp 950 >nul
echo ==================================================
echo   Studio0808 VoxCPM 語音合成工作站 (V20260614)
echo   PyInstaller 一鍵打包腳本 (方案 A - 排除模型檔)
echo ==================================================
echo.

:: 檢查是否安裝 PyInstaller
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 未偵測到 PyInstaller，正在為您安裝 PyInstaller...
    pip install pyinstaller
) else (
    echo [資訊] PyInstaller 已安裝，準備開始打包...
)

echo.
echo 正在清除舊的編譯快取與暫存目錄...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo ==================================================
echo [開始打包] 正在呼叫 PyInstaller 進行打包作業...
echo [提示] 此過程會分析 PyTorch, torchaudio, CustomTkinter 
echo        等巨大依賴庫，通常需要 2 至 5 分鐘，請耐心等候。
echo ==================================================
echo.

pyinstaller --clean -y --name="Studio0808_VoxCPM" --noconsole --icon="app.ico" --add-data="assets;assets" --collect-all="customtkinter" --collect-all="PIL" --collect-all="soundfile" --collect-all="sounddevice" --collect-all="torchaudio" --hidden-import="voxcpm" --hidden-import="modelscope" --hidden-import="huggingface_hub" Studio0808_VoxCPM.py

echo.
if %errorlevel% equ 0 (
    echo ==================================================
    echo [打包成功] 程式已順利打包完成！
    echo.
    echo 輸出資料夾：dist\Studio0808_VoxCPM\
    echo 主要主程式：dist\Studio0808_VoxCPM\Studio0808_VoxCPM.exe
    echo.
    echo 【部署與分發說明】：
    echo 1. 請將整個 `dist\Studio0808_VoxCPM\` 資料夾壓縮分發給使用者。
    echo 2. `models/` 與 `outputs/` 目錄已成功排除，節省了 4.6GB 體積。
    echo 3. 使用者點擊開啟 exe 後，程式會自動引導其下載與安裝模型。
    echo ==================================================
) else (
    echo ==================================================
    echo [錯誤] PyInstaller 打包過程中發生異常，請檢查上方錯誤訊息。
    echo ==================================================
)
echo.
pause
