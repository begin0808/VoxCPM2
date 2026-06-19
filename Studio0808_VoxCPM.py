import os
import sys
import random

# Force UTF-8 encoding for stdout/stderr in Windows to prevent cp950 codec encode crashes with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Set HF_ENDPOINT to hf-mirror for extremely fast downloads in Asia
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Monkey-patch torch.jit.script in packaged EXE to bypass "TorchScript requires source access" errors
if getattr(sys, 'frozen', False):
    try:
        import torch
        def dummy_script_method(fn, _rcb=None):
            return fn
        def dummy_script(obj, optimize=True, _frames_up=0, _rcb=None):
            return obj
        torch.jit.script_method = dummy_script_method
        torch.jit.script = dummy_script
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import soundfile as sf
import sounddevice as sd
import numpy as np
from PIL import Image

# Try importing CustomTkinter
try:
    import customtkinter as ctk
except ImportError:
    print("未安裝 CustomTkinter，請先執行 pip install customtkinter")
    sys.exit(1)

# Global variables
VOXCPM_MODEL = None
PLAYBACK_THREAD = None
PLAYBACK_STOP_EVENT = threading.Event()
IS_PLAYING = False

class SoundPlayer:
    @staticmethod
    def play_audio(filepath, stop_event, on_finished):
        global IS_PLAYING
        try:
            data, fs = sf.read(filepath)
            IS_PLAYING = True
            
            # Play using sounddevice
            sd.play(data, fs)
            
            # Wait until playback finishes or stop event is set
            chunk_size = int(fs * 0.1) # Check every 100ms
            position = 0
            while position < len(data) and not stop_event.is_set():
                time.sleep(0.1)
                position += chunk_size
            
            sd.stop()
        except Exception as e:
            print(f"播放音訊出錯: {e}")
        finally:
            IS_PLAYING = False
            on_finished()

class UILogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def log(self, msg):
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, f"{time.strftime('%H:%M:%S')} | {msg}\n")
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        except Exception as e:
            print(f"Log 寫入失敗: {e}")
        
        # Safe console print to prevent cp950 crashes
        try:
            enc = sys.stdout.encoding or 'utf-8'
            safe_msg = msg.encode(enc, errors='replace').decode(enc)
            print(safe_msg)
        except Exception:
            try:
                print(msg.encode('ascii', errors='ignore').decode('ascii'))
            except Exception:
                pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Variables
        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
            self.resource_dir = sys._MEIPASS
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
            self.resource_dir = os.path.dirname(os.path.abspath(__file__))

        # Configure window
        self.title("Studio0808 語音合成工作站 (V20260619)")
        self.geometry("1100x750")
        self.minsize(950, 680)
        
        # Set Application Icon & Taskbar ID on Windows
        try:
            import ctypes
            myappid = 'studio0808.voxcpm.app.v20260619'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

        try:
            icon_path = os.path.join(self.resource_dir, "app.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
                self.iconbitmap(default=icon_path)
                self.wm_iconbitmap(icon_path)
        except Exception as e:
            print(f"Warning: Could not load application icon app.ico. Error: {e}")
        
        # Center Window
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = int((ws/2) - (1100/2))
        y = int((hs/2) - (750/2) - 40)
        self.geometry(f"1100x750+{x}+{y}")
        
        # Maximize the window after mapping to prevent issues on Windows
        self.after(100, lambda: self.state('zoomed'))

        # Set theme
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Fonts
        self.font_title = ("Microsoft JhengHei UI", 20, "bold")
        self.font_header = ("Microsoft JhengHei UI", 16, "bold")
        self.font_ui = ("Microsoft JhengHei UI", 13)
        self.font_bold = ("Microsoft JhengHei UI", 13, "bold")
        self.font_mono = ("Microsoft JhengHei UI", 12)

        # Variables
        self.model_dir_var = ctk.StringVar(value=os.path.join(self.script_dir, "models", "VoxCPM2"))
        self.output_dir_var = ctk.StringVar(value=os.path.join(self.script_dir, "outputs"))
        self.downloader_source_var = ctk.StringVar(value="Hugging Face 官方 (推薦)")
        self.download_cancelled_event = threading.Event()
        self.is_downloading = False
        
        # Inference variables dictionaries (independent per view)
        self.cfg_scale_vars = {
            "design": ctk.DoubleVar(value=2.0),
            "clone": ctk.DoubleVar(value=2.0),
            "ultimate": ctk.DoubleVar(value=2.0)
        }
        self.timesteps_vars = {
            "design": ctk.IntVar(value=10),
            "clone": ctk.IntVar(value=10),
            "ultimate": ctk.IntVar(value=10)
        }
        self.speed_rate_vars = {
            "design": ctk.DoubleVar(value=1.0),
            "clone": ctk.DoubleVar(value=1.0),
            "ultimate": ctk.DoubleVar(value=1.0)
        }
        self.chk_norm_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        self.chk_denoise_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        
        # Seed variables
        self.chk_seed_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        self.seed_vars = {
            "design": ctk.StringVar(value=""),
            "clone": ctk.StringVar(value=""),
            "ultimate": ctk.StringVar(value="")
        }
        
        # Last generated audio
        self.last_generated_audio = None
        
        # Play buttons per view (to fix enable/disable across tabs)
        self.play_buttons = {}
        
        # Recording state variables
        self.is_recording = False
        self.recorded_frames = []
        self.recording_stream = None
        self.record_prompt_text = "這是現場即時錄音，可以用在聲音複製功能，看看效果有多逼真！"
        self.recorded_filepath = None
        self.recorded_play_thread = None
        self.recorded_play_stop_event = threading.Event()
        self.is_playing_recorded = False
        self.recording_device_var = ctk.StringVar()
        self.vis_buffer = np.zeros(4000)
        
        # Create UI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#181A1F")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1) # Spacer

        # Navigation Buttons
        self.nav_buttons = []
        self.nav_images = {}
        icons_dir = os.path.join(self.resource_dir, "assets", "icons")
        nav_items = [
            ("home", " 首頁", "#1E88E5", "home.png"),
            ("record", " 即時錄音", "#E91E63", "realtime_vc.png"),
            ("design", " 語音設計", "#9C27B0", "tts.png"),
            ("clone", " 聲音複製", "#4CAF50", "clone.png"),
            ("ultimate", " 極限複製", "#FF9800", "audio.png"),
            ("settings", " 系統設定", "#607D8B", "toolbox.png")
        ]

        for i, (name, label, color, icon_file) in enumerate(nav_items, start=0):
            img = None
            icon_path = os.path.join(icons_dir, icon_file)
            if os.path.exists(icon_path):
                img = ctk.CTkImage(light_image=Image.open(icon_path), size=(22, 22))
                self.nav_images[name] = img

            btn = ctk.CTkButton(
                self.sidebar_frame, 
                text=label, 
                font=self.font_bold, 
                image=img,
                compound="left",
                height=45,
                border_spacing=10,
                fg_color="transparent", 
                text_color=("gray20", "gray85"),
                hover_color=("#2A2D35"),
                anchor="w",
                command=lambda n=name: self.select_view(n)
            )
            pady_val = (20, 2) if i == 0 else 2
            btn.grid(row=i, column=0, padx=10, pady=pady_val, sticky="ew")
            self.nav_buttons.append((name, btn, color))

        # 2. Main Content Area
        self.content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=(15, 5))
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1) # Form area takes all height
        
        # 3. persistent Bottom Log Frame
        self.log_frame = ctk.CTkFrame(self, height=85, corner_radius=8, fg_color="#111216")
        self.log_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(1, weight=1)
        
        self.progress_bar = ctk.CTkProgressBar(self.log_frame, mode="indeterminate", height=6, fg_color="#1A1D24", progress_color="#E91E63")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(5, 2))
        self.progress_bar.set(0)
        
        self.log_box = ctk.CTkTextbox(self.log_frame, font=self.font_mono, fg_color="#000000", text_color="#FFFFFF", state="disabled")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 6))
        
        self.logger = UILogger(self.log_box)

        # Initialize Views
        self.views = {}
        self.create_views()
        
        # Select Default View
        self.select_view("home")
        
        # Check GPU status
        self.check_gpu()

    def check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.log(f"系統偵測到 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                self.logger.log(f"CUDA 版本: {torch.version.cuda} | PyTorch 版本: {torch.__version__}")
            else:
                self.logger.log("⚠️ 系統未偵測到 CUDA。將使用 CPU 模式（速度會慢上數十倍）。")
        except Exception as e:
            self.logger.log(f"偵測 GPU 時發生錯誤: {e}")
        self.logger.log("📢 提示：第一次使用時，請記得先至「系統設定」下載並部署官方模型權重。")

    def create_gradient_text_image(self, text, font_size=24, start_color=(233, 30, 99), end_color=(255, 152, 0)):
        from PIL import Image, ImageDraw, ImageFont
        font = None
        font_paths = [
            "msjhbd.ttc",
            "msjh.ttc",
            "C:\\Windows\\Fonts\\msjhbd.ttc",
            "C:\\Windows\\Fonts\\msjh.ttc",
            "arialbd.ttf",
            "arial.ttf"
        ]
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
            
        try:
            left, top, right, bottom = font.getbbox(text)
            w = right - left + 10
            h = bottom - top + 10
            offset_y = -top + 5
            offset_x = -left + 5
        except AttributeError:
            w, h = font.getsize(text)
            w += 10
            h += 10
            offset_x = 5
            offset_y = 5

        mask = Image.new("L", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.text((offset_x, offset_y), text, fill=255, font=font)
        
        gradient = Image.new("RGBA", (w, h))
        for x in range(w):
            factor = x / max(1, w - 1)
            r = int(start_color[0] + (end_color[0] - start_color[0]) * factor)
            g = int(start_color[1] + (end_color[1] - start_color[1]) * factor)
            b = int(start_color[2] + (end_color[2] - start_color[2]) * factor)
            for y in range(h):
                gradient.putpixel((x, y), (r, g, b, 255))
                
        gradient.putalpha(mask)
        return gradient

    def create_views(self):
        # We create panels inside content_frame
        # 1. Home View
        view_home = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["home"] = view_home
        
        # Title Row (No smile icon, normal text title)
        title_row = ctk.CTkFrame(view_home, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 5))
        
        lbl_title = ctk.CTkLabel(title_row, text="歡迎使用 Studio0808 語音合成工作站", font=self.font_title, anchor="w")
        lbl_title.pack(side="left")
        
        # Standard container (No scrollbar, spacing reduced)
        home_scroll = ctk.CTkFrame(view_home, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        home_scroll.pack(fill="both", expand=True, pady=(5, 5))
        
        # 1. Intro text
        intro_label = ctk.CTkLabel(
            home_scroll, 
            text="本工具是基於 OpenBMB 最新開源的語音合成模型 VoxCPM2 所開發的獨立測試程式。", 
            font=self.font_bold, 
            text_color="gray85",
            anchor="w",
            justify="left"
        )
        intro_label.pack(fill="x", padx=15, pady=(10, 5))
        
        # 首次使用提示卡片 (First-time user notice card)
        first_time_notice = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2196F3")
        first_time_notice.pack(fill="x", padx=15, pady=4, ipady=2)
        
        lbl_notice = ctk.CTkLabel(
            first_time_notice,
            text="💡 首次使用提示：本工具為全離線運行。若您是第一次開啟本程式，請務必先前往「系統設定」分頁下載並檢查 VoxCPM2 官方模型權重（共 7 個檔案，約 4.6GB），下載完成後即可開始使用語音合成與複製功能。",
            font=self.font_bold,
            text_color="#64B5F6",
            justify="left",
            anchor="w",
            wraplength=800
        )
        lbl_notice.pack(fill="x", padx=15, pady=6)
        
        # Highlights Card
        highlight_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        highlight_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_hl = ctk.CTkFrame(highlight_card, fg_color="transparent")
        header_hl.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_bulb = os.path.join(self.resource_dir, "assets", "icons", "bulb.png")
        if os.path.exists(icon_path_bulb):
            img_bulb = ctk.CTkImage(light_image=Image.open(icon_path_bulb), size=(20, 20))
            lbl_icon_hl = ctk.CTkLabel(header_hl, image=img_bulb, text="")
            lbl_icon_hl.pack(side="left", padx=(0, 8))
            
        lbl_title_hl = ctk.CTkLabel(header_hl, text="主要特色與亮點", font=self.font_header, text_color="#2196F3")
        lbl_title_hl.pack(side="left")
        
        bullet_text = (
            "• 離線免除 discrete tokenization，直接產生高品質連續音訊表現。\n"
            "• 【語音設計】：免參考語音，僅用語音特徵描述（性別、語氣、年齡等）即可無中生有生成聲音。\n"
            "• 【聲音複製】：支援 3~10 秒極短音訊即可模仿對方音色與情緒。\n"
            "• 【極限複製】：配合參考逐字稿，達成無瑕疵、極致真實的語氣接續合成。\n"
            "• 支援高達 30 種語言合成（包含粵語、閩南語等漢語方言）。\n"
            "• 輸出品質直接高達 48kHz 錄音室等級音質。"
        )
        lbl_content_hl = ctk.CTkLabel(
            highlight_card, 
            text=bullet_text, 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w"
        )
        lbl_content_hl.pack(fill="x", padx=25, pady=(2, 6))
        
        # Hardware Card
        hw_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        hw_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_hw = ctk.CTkFrame(hw_card, fg_color="transparent")
        header_hw.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_hammer = os.path.join(self.resource_dir, "assets", "icons", "toolbox.png")
        if os.path.exists(icon_path_hammer):
            img_hammer = ctk.CTkImage(light_image=Image.open(icon_path_hammer), size=(20, 20))
            lbl_icon_hw = ctk.CTkLabel(header_hw, image=img_hammer, text="")
            lbl_icon_hw.pack(side="left", padx=(0, 8))
            
        lbl_title_hw = ctk.CTkLabel(header_hw, text="硬體建議", font=self.font_header, text_color="#4CAF50")
        lbl_title_hw.pack(side="left")
        
        hw_text = "本模型參數量為 2B (約 20億)，推論需要約 3GB~5GB VRAM。強烈建議配備 NVIDIA 顯示卡以獲得秒級生成體驗。"
        lbl_content_hw = ctk.CTkLabel(
            hw_card, 
            text=hw_text, 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w",
            wraplength=800
        )
        lbl_content_hw.pack(fill="x", padx=25, pady=(2, 6))
        
        # License Card (Warning Style)
        license_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        license_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_lic = ctk.CTkFrame(license_card, fg_color="transparent")
        header_lic.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_warning = os.path.join(self.resource_dir, "assets", "icons", "warning.png")
        if os.path.exists(icon_path_warning):
            img_warning = ctk.CTkImage(light_image=Image.open(icon_path_warning), size=(20, 20))
            lbl_icon_lic = ctk.CTkLabel(header_lic, image=img_warning, text="")
            lbl_icon_lic.pack(side="left", padx=(0, 8))
            
        lbl_title_lic = ctk.CTkLabel(header_lic, text="開源授權與版權宣告 (Open Source License & Credits)", font=self.font_header, text_color="#FF5252")
        lbl_title_lic.pack(side="left")
        
        lic_text = (
            "• 本專案使用之 VoxCPM2 模型及其核心推論代碼源自開源專案 VoxCPM (https://github.com/OpenBMB/VoxCPM)。\n"
            "• VoxCPM 原專案採用 Apache License 2.0 授權協議。\n"
            "• 版權所有 © 2026 OpenBMB。本工具僅供個人學術研究、測試與開發評估之用，請遵守相關法律法規，勿用於非法或未授權的語音合成傳播。"
        )
        lbl_content_lic = ctk.CTkLabel(
            license_card, 
            text=lic_text, 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w",
            wraplength=800
        )
        lbl_content_lic.pack(fill="x", padx=25, pady=(5, 10))
        
        # 2. Design View
        view_design = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["design"] = view_design
        
        ctk.CTkLabel(view_design, text="語音設計 (Voice Design)", font=self.font_title, anchor="w").pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(view_design, text="無須上傳參考音訊。直接在文字開頭加上括號與音色特徵描述即可生成特定聲線。\n例如：(A gentle young female voice, smiling) 哈囉，歡迎來到語音合成工作站！\n💡 批次模式：每行一句，按「批次合成」可逐行依序全部生成。", font=self.font_ui, text_color="gray", anchor="w").pack(fill="x", pady=(0, 8))
        
        # Container for text input and presets
        design_input_container = ctk.CTkFrame(view_design, fg_color="transparent")
        design_input_container.pack(fill="x", pady=4)
        design_input_container.grid_columnconfigure(0, weight=1)
        design_input_container.grid_columnconfigure(1, weight=0)
        
        left_side = ctk.CTkFrame(design_input_container, fg_color="transparent")
        left_side.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(left_side, text="目標合成文字 (請包含描述，批次模式每行一句):", font=self.font_bold, anchor="w").pack(fill="x")
        self.txt_design = ctk.CTkTextbox(left_side, height=180, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.txt_design.insert("0.0", "(A young woman, gentle and sweet voice) 哈囉！這是使用語音設計無中生有出來的聲音。")
        self.txt_design.pack(fill="both", expand=True, pady=(2, 0))
        
        right_side = ctk.CTkFrame(design_input_container, fg_color="#181A1F", border_width=1, border_color="#2A2D35", width=260)
        right_side.grid(row=0, column=1, sticky="nsew")
        right_side.grid_propagate(False)
        
        ctk.CTkLabel(right_side, text=" 語音特徵快速範本", font=self.font_bold, text_color="#9C27B0", anchor="w").pack(pady=(8, 2), padx=10, fill="x")
        
        self.preset_menus = {}
        
        self.preset_menus["tw_mandarin"] = ctk.CTkOptionMenu(
            right_side,
            values=["臺灣老爺爺", "臺灣老奶奶", "臺灣年輕男生", "臺灣年輕女生", "臺灣小男孩", "臺灣小女孩"],
            command=lambda r: self._apply_voice_preset("tw_mandarin", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#2196F3",
            button_hover_color="#1976D2"
        )
        self.preset_menus["tw_mandarin"].set("臺灣國語 (選擇音色)")
        self.preset_menus["tw_mandarin"].pack(pady=3, padx=15, fill="x")
        
        
        self.preset_menus["japanese"] = ctk.CTkOptionMenu(
            right_side,
            values=["老爺爺", "老奶奶", "年輕男生", "年輕女生", "小男孩", "小女孩"],
            command=lambda r: self._apply_voice_preset("japanese", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#FF9800",
            button_hover_color="#F57C00"
        )
        self.preset_menus["japanese"].set("日本語 (選擇音色)")
        self.preset_menus["japanese"].pack(pady=3, padx=15, fill="x")
        
        self.preset_menus["us_english"] = ctk.CTkOptionMenu(
            right_side,
            values=["老爺爺", "老奶奶", "年輕男生", "年輕女生", "小男孩", "小女孩"],
            command=lambda r: self._apply_voice_preset("us_english", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#9C27B0",
            button_hover_color="#7B1FA2"
        )
        self.preset_menus["us_english"].set("美國英語 (選擇音色)")
        self.preset_menus["us_english"].pack(pady=3, padx=15, fill="x")
        
        self.preset_menus["korean"] = ctk.CTkOptionMenu(
            right_side,
            values=["老爺爺", "老奶奶", "年輕男生", "年輕女生", "小男孩", "小女孩"],
            command=lambda r: self._apply_voice_preset("korean", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#607D8B",
            button_hover_color="#455A64"
        )
        self.preset_menus["korean"].set("韓語 (選擇音色)")
        self.preset_menus["korean"].pack(pady=(3, 8), padx=15, fill="x")
        
        self.create_inference_params_ui(view_design, "design")
        self.create_action_buttons_ui(view_design, "design")

        # 3. Clone View
        view_clone = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["clone"] = view_clone
        
        ctk.CTkLabel(view_clone, text="聲音複製 (Voice Clone)", font=self.font_title, anchor="w").pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(view_clone, text="只需 3-10 秒乾淨人聲音訊，即可完美複製對方的音色、情感與說話節奏。", font=self.font_ui, text_color="gray", anchor="w").pack(fill="x", pady=(0, 8))
        
        # File selector row
        row_file = ctk.CTkFrame(view_clone, fg_color="transparent")
        row_file.pack(fill="x", pady=3)
        ctk.CTkLabel(row_file, text="參考語音檔案 (WAV/MP3/FLAC):", font=self.font_bold, width=190, anchor="w").pack(side="left")
        self.entry_clone_ref = ctk.CTkEntry(row_file, font=self.font_ui, placeholder_text="請選取 3~10秒 的 WAV/MP3/FLAC 音檔...", fg_color="#181A1F")
        self.entry_clone_ref.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row_file, text="瀏覽...", font=self.font_ui, width=80, command=self.browse_clone_ref).pack(side="left")
        
        ctk.CTkLabel(view_clone, text="目標合成文字:", font=self.font_bold, anchor="w").pack(fill="x", pady=(6, 0))
        self.txt_clone = ctk.CTkTextbox(view_clone, height=130, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.txt_clone.insert("0.0", "你好，這是我複製你的音色所說出來的一段話。聽起來效果如何？")
        self.txt_clone.pack(fill="x", pady=4)
        
        self.create_inference_params_ui(view_clone, "clone")
        self.create_action_buttons_ui(view_clone, "clone")

        # 4. Ultimate View
        view_ultimate = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["ultimate"] = view_ultimate
        
        ctk.CTkLabel(view_ultimate, text="極限複製 (Ultimate Clone)", font=self.font_title, anchor="w").pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(view_ultimate, text="同時提供參考音檔及其對應的逐字稿。模型會無縫延續該段語氣，達成最高品質與真實度的模仿。", font=self.font_ui, text_color="gray", anchor="w").pack(fill="x", pady=(0, 8))
        
        # File selector row
        row_file2 = ctk.CTkFrame(view_ultimate, fg_color="transparent")
        row_file2.pack(fill="x", pady=3)
        ctk.CTkLabel(row_file2, text="參考語音檔案 (WAV/MP3/FLAC):", font=self.font_bold, width=190, anchor="w").pack(side="left")
        self.entry_ult_ref = ctk.CTkEntry(row_file2, font=self.font_ui, placeholder_text="請選取 WAV/MP3/FLAC 參考音檔...", fg_color="#181A1F")
        self.entry_ult_ref.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row_file2, text="瀏覽...", font=self.font_ui, width=80, command=self.browse_ult_ref).pack(side="left")
        
        # Prompt text row
        row_prompt = ctk.CTkFrame(view_ultimate, fg_color="transparent")
        row_prompt.pack(fill="x", pady=3)
        ctk.CTkLabel(row_prompt, text="參考語意逐字稿 (必填):", font=self.font_bold, width=150, anchor="w").pack(side="left")
        self.entry_ult_prompt = ctk.CTkEntry(row_prompt, font=self.font_ui, placeholder_text="輸入參考音檔中說出的完整逐字內容...", fg_color="#181A1F")
        self.entry_ult_prompt.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(view_ultimate, text="目標合成文字 (將接續在參考語音後面說):", font=self.font_bold, anchor="w").pack(fill="x", pady=(6, 0))
        self.txt_ultimate = ctk.CTkTextbox(view_ultimate, height=120, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.txt_ultimate.insert("0.0", "接下去說這句話，完全保留你原本說話的情緒、呼吸起伏與語調。")
        self.txt_ultimate.pack(fill="x", pady=4)
        
        self.create_inference_params_ui(view_ultimate, "ultimate")
        self.create_action_buttons_ui(view_ultimate, "ultimate")

        # 5. Settings View
        view_settings = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["settings"] = view_settings
        
        ctk.CTkLabel(view_settings, text="系統設定與模型管理", font=self.font_title, anchor="w").pack(fill="x", pady=(0, 15))
        
        # Model path setting
        row_path = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_path.pack(fill="x", pady=8)
        ctk.CTkLabel(row_path, text="本機模型儲存路徑:", font=self.font_bold, width=150, anchor="w").pack(side="left")
        self.entry_model_dir = ctk.CTkEntry(row_path, textvariable=self.model_dir_var, font=self.font_ui, fg_color="#181A1F")
        self.entry_model_dir.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row_path, text="選擇資料夾...", font=self.font_ui, width=120, command=self.browse_model_dir).pack(side="left")

        # Output path setting
        row_out_path = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_out_path.pack(fill="x", pady=8)
        ctk.CTkLabel(row_out_path, text="生成語音儲存路徑:", font=self.font_bold, width=150, anchor="w").pack(side="left")
        self.entry_output_dir = ctk.CTkEntry(row_out_path, textvariable=self.output_dir_var, font=self.font_ui, fg_color="#181A1F")
        self.entry_output_dir.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row_out_path, text="選擇資料夾...", font=self.font_ui, width=120, command=self.browse_output_dir).pack(side="left")

        # Separator line
        ctk.CTkFrame(view_settings, height=2, fg_color="#2A2D35").pack(fill="x", pady=15)

        # Download Panel
        ctk.CTkLabel(view_settings, text="VoxCPM2 官方模型下載與修補", font=self.font_header, anchor="w").pack(fill="x", pady=(0, 5))
        
        info_text = (
            "本程式為全離線運作。首次使用請下載 OpenBMB/VoxCPM2 模型權重，下載完成後會自動部署於本機路徑。\n"
            "⚠️ 本模型共包含下列 7 個必要檔案，總計約 4.63 GB，下載過程需要一些時間，請耐心等候：\n"
            "   1. config.json (4.2 KB)           2. special_tokens_map.json (1.6 KB)    3. tokenization_voxcpm2.py (2.8 KB)\n"
            "   4. tokenizer_config.json (4.9 KB)  5. tokenizer.json (3.5 MB)             6. audiovae.pth (359.5 MB)\n"
            "   7. model.safetensors (4.27 GB)"
        )
        ctk.CTkLabel(view_settings, text=info_text, font=self.font_ui, text_color="gray", justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        row_dl = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_dl.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row_dl, text="下載伺服器來源:", font=self.font_ui).pack(side="left", padx=(0, 10))
        ctk.CTkOptionMenu(
            row_dl, 
            variable=self.downloader_source_var, 
            values=[
                "Hugging Face 官方 (推薦)", 
                "Hugging Face 鏡像", 
                "ModelScope"
            ], 
            font=self.font_ui, 
            dropdown_font=self.font_ui, 
            width=220
        ).pack(side="left")
        
        self.btn_download = ctk.CTkButton(row_dl, text="開始下載 / 檢查模型", font=self.font_bold, fg_color="#9C27B0", hover_color="#7B1FA2", width=180, command=self.start_model_download)
        self.btn_download.pack(side="left", padx=15)

        self.lbl_dl_status = ctk.CTkLabel(view_settings, text="", font=self.font_ui, text_color="gray85", anchor="w", justify="left")
        self.lbl_dl_status.pack(fill="x", pady=(5, 0))

        # 1.5. Record View
        view_record = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["record"] = view_record
        
        ctk.CTkLabel(view_record, text="即時錄音測試 (Live Recording Test)", font=self.font_title, anchor="w").pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(view_record, text="在此錄製您的聲音以測試複製功能。系統將使用您的錄音作為「聲音複製」與「極限複製」的參考音訊。", font=self.font_ui, text_color="gray", anchor="w").pack(fill="x", pady=(0, 8))
        
        # Recording device selection frame
        row_device = ctk.CTkFrame(view_record, fg_color="transparent")
        row_device.pack(fill="x", pady=5)
        ctk.CTkLabel(row_device, text="選擇錄音輸入設備 (麥克風):", font=self.font_bold, width=200, anchor="w").pack(side="left")
        
        self.menu_devices = ctk.CTkOptionMenu(
            row_device,
            variable=self.recording_device_var,
            values=["預設輸入裝置"],
            font=self.font_ui,
            dropdown_font=self.font_ui,
            width=300
        )
        self.menu_devices.pack(side="left", padx=5)
        
        btn_refresh_devices = ctk.CTkButton(
            row_device,
            text="🔄 重新整理",
            font=self.font_bold,
            width=100,
            command=self.populate_recording_devices
        )
        btn_refresh_devices.pack(side="left", padx=5)
        
        # Prompt text card (Reading card)
        prompt_card = ctk.CTkFrame(view_record, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        prompt_card.pack(fill="x", pady=10, ipady=10)
        
        ctk.CTkLabel(prompt_card, text="請看著下方文本，並以正常音量與速度朗讀：", font=self.font_bold, text_color="#E91E63").pack(pady=(10, 5), padx=15, anchor="w")
        
        lbl_prompt_text = ctk.CTkLabel(
            prompt_card,
            text=f"「{self.record_prompt_text}」",
            font=("Microsoft JhengHei UI", 16, "bold"),
            text_color="#FFFFFF",
            justify="center",
            wraplength=800
        )
        lbl_prompt_text.pack(pady=15, padx=20, fill="x")
        
        # Note about audio length
        ctk.CTkLabel(
            prompt_card,
            text="💡 提示：本文字長度約 5~7 秒，是最適合聲音複製的黃金長度（建議控制在 3~10 秒內）。",
            font=self.font_ui,
            text_color="gray"
        ).pack(pady=(0, 10))
        
        # Recording status and control buttons
        control_frame = ctk.CTkFrame(view_record, fg_color="transparent")
        control_frame.pack(fill="x", pady=10)
        
        self.btn_record = ctk.CTkButton(
            control_frame,
            text="🔴 開始錄音",
            font=self.font_bold,
            height=45,
            width=180,
            fg_color="#F44336",
            hover_color="#D32F2F",
            command=self.toggle_recording
        )
        self.btn_record.pack(side="left", padx=(0, 10))
        
        # Real-time voice waveform canvas
        self.wave_canvas = tk.Canvas(control_frame, width=250, height=45, bg="#111216", highlightthickness=1, highlightbackground="#2A2D35")
        self.wave_canvas.pack(side="left", padx=10)
        # Draw initial flat line
        self.wave_canvas.create_line(0, 22, 250, 22, fill="#555555", tags="wave", width=1)
        
        self.lbl_record_status = ctk.CTkLabel(
            control_frame,
            text="就緒。請點選「開始錄音」以開始讀稿。",
            font=self.font_bold,
            text_color="gray85"
        )
        self.lbl_record_status.pack(side="left", padx=10)
        
        # Playback and Apply frame
        self.recorded_actions_frame = ctk.CTkFrame(view_record, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.recorded_actions_frame.pack(fill="x", pady=10, ipady=5)
        
        ctk.CTkLabel(self.recorded_actions_frame, text="錄音後續作業：", font=self.font_bold).pack(anchor="w", padx=15, pady=(8, 4))
        
        actions_row = ctk.CTkFrame(self.recorded_actions_frame, fg_color="transparent")
        actions_row.pack(fill="x", padx=10, pady=5)
        
        self.btn_play_recorded = ctk.CTkButton(
            actions_row,
            text="▶ 播放錄音",
            font=self.font_bold,
            height=40,
            width=150,
            fg_color="#4CAF50",
            hover_color="#388E3C",
            state="disabled",
            command=self.toggle_recorded_playback
        )
        self.btn_play_recorded.pack(side="left", padx=5)
        
        self.btn_apply_clone = ctk.CTkButton(
            actions_row,
            text="👥 套用至 聲音複製",
            font=self.font_bold,
            height=40,
            width=180,
            fg_color="#2196F3",
            hover_color="#1976D2",
            state="disabled",
            command=self.use_recorded_for_clone
        )
        self.btn_apply_clone.pack(side="left", padx=5)
        
        self.btn_apply_ultimate = ctk.CTkButton(
            actions_row,
            text="👑 套用至 極限複製",
            font=self.font_bold,
            height=40,
            width=180,
            fg_color="#FF9800",
            hover_color="#F57C00",
            state="disabled",
            command=self.use_recorded_for_ultimate
        )
        self.btn_apply_ultimate.pack(side="left", padx=5)
        
        # Initialize recording device list
        self.populate_recording_devices()

    def create_inference_params_ui(self, parent_frame, func_type):
        # A tidy parameters subframe
        param_frame = ctk.CTkFrame(parent_frame, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        param_frame.pack(fill="x", pady=6, ipady=2)
        
        ctk.CTkLabel(param_frame, text="推論進階參數設定", font=self.font_header).pack(anchor="w", padx=15, pady=(6, 2))
        
        # Grid container inside param_frame to arrange two columns
        grid_container = ctk.CTkFrame(param_frame, fg_color="transparent")
        grid_container.pack(fill="x", padx=10, pady=(0, 4))
        grid_container.columnconfigure(0, weight=0)
        grid_container.columnconfigure(1, weight=1)
        
        # Left column frame
        left_col = ctk.CTkFrame(grid_container, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(5, 15))
        left_col.columnconfigure(2, weight=1)
        
        # Right column frame
        right_col = ctk.CTkFrame(grid_container, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(15, 5))
        right_col.columnconfigure(0, weight=1)

        # === Left Column Sliders ===
        # 1. CFG Scale (Row 0)
        ctk.CTkLabel(left_col, text="引導係數 (CFG):", font=self.font_bold).grid(row=0, column=0, padx=(0, 5), pady=3, sticky="e")
        
        cfg_container = ctk.CTkFrame(left_col, fg_color="transparent")
        cfg_container.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        
        lbl_cfg_val = ctk.CTkLabel(cfg_container, text=f"{self.cfg_scale_vars[func_type].get():.1f}", font=self.font_bold, width=35)
        def _update_cfg(v):
            val = round(v * 2) / 2
            lbl_cfg_val.configure(text=f"{val:.1f}")
            self.cfg_scale_vars[func_type].set(val)
        
        ctk.CTkSlider(cfg_container, from_=1.0, to=5.0, number_of_steps=8, variable=self.cfg_scale_vars[func_type], command=_update_cfg, width=150).pack(side="left")
        lbl_cfg_val.pack(side="left", padx=(5, 0))
        
        ctk.CTkLabel(left_col, text="越小音質越好但描述度越弱，建議 2.0-3.0", font=self.font_ui, text_color="gray").grid(row=0, column=2, padx=10, pady=3, sticky="w")
        
        # 2. Timesteps (Row 1)
        ctk.CTkLabel(left_col, text="去噪步數 (Steps):", font=self.font_bold).grid(row=1, column=0, padx=(0, 5), pady=3, sticky="e")
        
        step_container = ctk.CTkFrame(left_col, fg_color="transparent")
        step_container.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        
        lbl_step_val = ctk.CTkLabel(step_container, text=f"{self.timesteps_vars[func_type].get()}", font=self.font_bold, width=35)
        def _update_step(v):
            val = int(v)
            lbl_step_val.configure(text=f"{val}")
            self.timesteps_vars[func_type].set(val)
            
        ctk.CTkSlider(step_container, from_=5, to=30, number_of_steps=25, variable=self.timesteps_vars[func_type], command=_update_step, width=150).pack(side="left")
        lbl_step_val.pack(side="left", padx=(5, 0))
        
        ctk.CTkLabel(left_col, text="步數越多生成越精細但耗時增加，一般設為 10-15 即可", font=self.font_ui, text_color="gray").grid(row=1, column=2, padx=10, pady=3, sticky="w")
        
        # 3. Speed Rate (Row 2)
        ctk.CTkLabel(left_col, text="語速設定 (Speed):", font=self.font_bold).grid(row=2, column=0, padx=(0, 5), pady=3, sticky="e")
        
        speed_container = ctk.CTkFrame(left_col, fg_color="transparent")
        speed_container.grid(row=2, column=1, padx=5, pady=3, sticky="w")
        
        lbl_speed_val = ctk.CTkLabel(speed_container, text=f"{self.speed_rate_vars[func_type].get():.1f}x", font=self.font_bold, width=35)
        def _update_speed(v):
            val = round(v, 1)
            lbl_speed_val.configure(text=f"{val:.1f}x")
            self.speed_rate_vars[func_type].set(val)
            
        ctk.CTkSlider(speed_container, from_=0.5, to=2.0, number_of_steps=15, variable=self.speed_rate_vars[func_type], command=_update_speed, width=150).pack(side="left")
        lbl_speed_val.pack(side="left", padx=(5, 0))
        
        ctk.CTkLabel(left_col, text="調整生成語音速度 (0.5x 低速 - 2.0x 快速，預設 1.0x)", font=self.font_ui, text_color="gray").grid(row=2, column=2, padx=10, pady=3, sticky="w")

        # === Right Column Parameters ===
        # 1. Text Normalization (Row 0)
        chk_norm = ctk.CTkCheckBox(right_col, text="文字標準化 (將數字/符號轉文字)", font=self.font_bold, variable=self.chk_norm_vars[func_type], checkbox_width=16, checkbox_height=16)
        chk_norm.grid(row=0, column=0, padx=(15, 0), pady=(12, 6), sticky="w")
        
        # 2. Denoise Reference (Row 1)
        chk_denoise = ctk.CTkCheckBox(right_col, text="參考音訊自動降噪", font=self.font_bold, variable=self.chk_denoise_vars[func_type], checkbox_width=16, checkbox_height=16)
        chk_denoise.grid(row=1, column=0, padx=(15, 0), pady=6, sticky="w")
        
        # 3. Fixed Seed Frame (Row 2)
        seed_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        seed_frame.grid(row=2, column=0, padx=(15, 0), pady=(6, 12), sticky="w")
        
        chk_seed = ctk.CTkCheckBox(
            seed_frame, 
            text="使用固定隨機種子 (固定音色)", 
            font=self.font_bold, 
            variable=self.chk_seed_vars[func_type], 
            checkbox_width=16, 
            checkbox_height=16,
            command=lambda f=func_type: self._toggle_seed_entry(f)
        )
        chk_seed.pack(side="left")
        
        if not hasattr(self, "entry_seeds"):
            self.entry_seeds = {}
            
        entry_seed = ctk.CTkEntry(
            seed_frame, 
            textvariable=self.seed_vars[func_type], 
            font=self.font_mono, 
            width=90, 
            height=24,
            fg_color="#181A1F"
        )
        entry_seed.pack(side="left", padx=(8, 0))
        self.entry_seeds[func_type] = entry_seed
        
        # Disable/Enable seed input based on checkbox initial value
        if not self.chk_seed_vars[func_type].get():
            entry_seed.configure(state="disabled")

    def create_action_buttons_ui(self, parent_frame, func_type):
        action_row = ctk.CTkFrame(parent_frame, fg_color="transparent")
        action_row.pack(fill="x", pady=6)
        
        # Start Inference
        btn_run = ctk.CTkButton(
            action_row, 
            text="🚀 開始語音合成", 
            font=self.font_bold, 
            height=40, 
            width=180, 
            fg_color="#E91E63", 
            hover_color="#C2185B",
            command=lambda f=func_type: self.run_inference(f)
        )
        btn_run.pack(side="left", padx=(0, 10))
        
        # Batch Synthesis button (only for design view)
        if func_type == "design":
            btn_batch = ctk.CTkButton(
                action_row, 
                text="📋 批次合成 (逐行)", 
                font=self.font_bold, 
                height=40, 
                width=180, 
                fg_color="#FF9800", 
                hover_color="#F57C00",
                command=self.run_batch_inference
            )
            btn_batch.pack(side="left", padx=(0, 10))
        
        # Audio Playback section
        play_frame = ctk.CTkFrame(action_row, fg_color="transparent")
        play_frame.pack(side="left", padx=10)
        
        btn_play = ctk.CTkButton(
            play_frame, 
            text="▶ 播放生成音訊", 
            font=self.font_bold, 
            height=40, 
            width=150, 
            fg_color="#4CAF50", 
            hover_color="#388E3C",
            state="disabled",
            command=self.toggle_playback
        )
        btn_play.pack(side="left", padx=5)
        self.play_buttons[func_type] = btn_play

        btn_open = ctk.CTkButton(
            action_row, 
            text="📂 開啟輸出資料夾", 
            font=self.font_bold, 
            height=40, 
            width=150, 
            fg_color="#607D8B", 
            hover_color="#455A64",
            command=self.open_output_folder
        )
        btn_open.pack(side="left", padx=10)

    # Browser helpers
    def browse_clone_ref(self):
        f = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.m4a")])
        if f: self.entry_clone_ref.delete(0, tk.END); self.entry_clone_ref.insert(0, f)

    def browse_ult_ref(self):
        f = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.m4a")])
        if f: self.entry_ult_ref.delete(0, tk.END); self.entry_ult_ref.insert(0, f)

    def browse_model_dir(self):
        d = filedialog.askdirectory()
        if d: self.model_dir_var.set(d)

    def browse_output_dir(self):
        d = filedialog.askdirectory()
        if d: self.output_dir_var.set(d)

    def select_view(self, name):
        # Update Nav bar selection colors
        for btn_name, btn, color in self.nav_buttons:
            if btn_name == name:
                btn.configure(fg_color=color, text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=("gray20", "gray85"))

        # Forget all views
        for view in self.views.values():
            view.pack_forget()

        # Render selected view
        if name in self.views:
            self.views[name].pack(fill="both", expand=True)

    # Inference thread logic
    def run_inference(self, func_type):
        model_dir = self.model_dir_var.get()
        # Verify model dir
        if not os.path.exists(os.path.join(model_dir, "config.json")) and not os.path.exists(os.path.join(model_dir, "pyproject.toml")):
            messagebox.showwarning("警告", "本機模型路徑未檢測到 config.json，可能尚未下載模型！\n請至「系統設定」分頁下載模型，或將本機模型路徑指向正確的資料夾。")
            return

        # Disable main window run buttons to prevent double entry
        self.logger.log("🔄 正在背景準備推論作業，請稍候...")
        
        # Start progress bar animation
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        # Determine params based on view
        if func_type == "design":
            text = self.txt_design.get("0.0", "end").strip()
            args = {
                "text": text, 
                "cfg_value": self.cfg_scale_vars["design"].get(), 
                "inference_timesteps": self.timesteps_vars["design"].get(),
                "normalize": self.chk_norm_vars["design"].get(),
                "denoise": self.chk_denoise_vars["design"].get()
            }
        elif func_type == "clone":
            text = self.txt_clone.get("0.0", "end").strip()
            ref_audio = self.entry_clone_ref.get().strip()
            if not ref_audio or not os.path.exists(ref_audio):
                messagebox.showerror("錯誤", "請提供有效的參考音檔路徑！")
                self._stop_progress_bar()
                return
            args = {
                "text": text, 
                "reference_wav_path": ref_audio, 
                "cfg_value": self.cfg_scale_vars["clone"].get(), 
                "inference_timesteps": self.timesteps_vars["clone"].get(),
                "normalize": self.chk_norm_vars["clone"].get(),
                "denoise": self.chk_denoise_vars["clone"].get()
            }
        elif func_type == "ultimate":
            text = self.txt_ultimate.get("0.0", "end").strip()
            ref_audio = self.entry_ult_ref.get().strip()
            prompt_text = self.entry_ult_prompt.get().strip()
            if not ref_audio or not os.path.exists(ref_audio):
                messagebox.showerror("錯誤", "請提供有效的參考音檔路徑！")
                self._stop_progress_bar()
                return
            if not prompt_text:
                messagebox.showerror("錯誤", "極限複製模式必須提供參考音訊的逐字稿！")
                self._stop_progress_bar()
                return
            args = {
                "text": text, 
                "prompt_wav_path": ref_audio, 
                "prompt_text": prompt_text, 
                "reference_wav_path": ref_audio,
                "cfg_value": self.cfg_scale_vars["ultimate"].get(),
                "inference_timesteps": self.timesteps_vars["ultimate"].get(),
                "normalize": self.chk_norm_vars["ultimate"].get(),
                "denoise": self.chk_denoise_vars["ultimate"].get()
            }
        
        # Check seed setting
        if self.chk_seed_vars[func_type].get():
            try:
                seed_val = int(self.seed_vars[func_type].get().strip())
                args["seed"] = seed_val
            except ValueError:
                messagebox.showerror("錯誤", "隨機種子必須是整數！")
                self._stop_progress_bar()
                return
                
        speed_rate = self.speed_rate_vars[func_type].get()

        # Start thread
        threading.Thread(target=self._infer_thread_run, args=(model_dir, args, speed_rate), daemon=True).start()

    def run_batch_inference(self):
        """Batch synthesis: split textbox by lines, generate each one sequentially."""
        raw_text = self.txt_design.get("0.0", "end").strip()
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        if not lines:
            messagebox.showwarning("警告", "文字框中沒有任何內容可供合成！")
            return
        
        if len(lines) < 2:
            # Only one line, just use normal single inference
            self.run_inference("design")
            return
        
        model_dir = self.model_dir_var.get()
        if not os.path.exists(os.path.join(model_dir, "config.json")) and not os.path.exists(os.path.join(model_dir, "pyproject.toml")):
            messagebox.showwarning("警告", "本機模型路徑未檢測到 config.json，可能尚未下載模型！\n請至「系統設定」分頁下載模型。")
            return
        
        self.logger.log(f"📋 批次合成模式啟動！共 {len(lines)} 句待處理。")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        cfg = self.cfg_scale_vars["design"].get()
        steps = self.timesteps_vars["design"].get()
        speed_rate = self.speed_rate_vars["design"].get()
        norm = self.chk_norm_vars["design"].get()
        denoise_audio = self.chk_denoise_vars["design"].get()
        
        # Check seed setting
        seed_val = None
        if self.chk_seed_vars["design"].get():
            try:
                seed_val = int(self.seed_vars["design"].get().strip())
            except ValueError:
                messagebox.showwarning("警告", "隨機種子必須是整數！")
                self._stop_progress_bar()
                return
                
        threading.Thread(target=self._batch_infer_thread_run, args=(model_dir, lines, cfg, steps, speed_rate, norm, denoise_audio, seed_val), daemon=True).start()

    def _batch_infer_thread_run(self, model_dir, lines, cfg, steps, speed_rate, norm, denoise_audio, seed_val):
        global VOXCPM_MODEL
        total = len(lines)
        success_count = 0
        fail_count = 0
        generated_files = []
        batch_start = time.time()
        
        try:
            # Lazy load model if not loaded
            if VOXCPM_MODEL is None:
                self.logger.log("🔄 正在初次初始化並載入 VoxCPM2 模型... (模型大小為 2B，第一次載入可能需要 20-40 秒)")
                from voxcpm import VoxCPM
                VOXCPM_MODEL = VoxCPM.from_pretrained(model_dir, load_denoiser=False, optimize=not getattr(sys, 'frozen', False))
                self.logger.log("✅ 模型成功加載至記憶體/顯示記憶體！")
            
            for i, line in enumerate(lines, 1):
                self.logger.log(f"")
                self.logger.log(f"━━━ 批次進度 [{i}/{total}] ━━━")
                self.logger.log(f"📝 文字: {line[:80]}{'...' if len(line) > 80 else ''}")
                
                # Apply seed
                if seed_val is not None:
                    self.set_global_seed(seed_val)
                    self.logger.log(f"🔒 使用固定種子 ID: {seed_val}")
                else:
                    line_seed = random.randint(0, 10000000)
                    self.set_global_seed(line_seed)
                    self.logger.log(f"🎲 使用隨機種子 ID: {line_seed} (可複製此 ID 並勾選「固定隨機種子」來固定音色)")
                
                try:
                    start_time = time.time()
                    wav = VOXCPM_MODEL.generate(
                        text=line,
                        cfg_value=cfg,
                        inference_timesteps=steps,
                        normalize=norm,
                        denoise=denoise_audio,
                    )
                    
                    if speed_rate != 1.0:
                        self.logger.log(f"⏳ 正在調整語速為 {speed_rate}x...")
                        import librosa
                        wav = librosa.effects.time_stretch(wav, rate=speed_rate)
                    
                    os.makedirs(self.output_dir_var.get(), exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"voxcpm_batch{i:02d}_{timestamp}.wav"
                    filepath = os.path.join(self.output_dir_var.get(), filename)
                    sf.write(filepath, wav, VOXCPM_MODEL.tts_model.sample_rate)
                    
                    elapsed = time.time() - start_time
                    self.logger.log(f"✅ 第 {i} 句完成！耗時: {elapsed:.2f} 秒 → {filename}")
                    
                    self.last_generated_audio = filepath
                    generated_files.append(filepath)
                    success_count += 1
                    
                except Exception as e:
                    self.logger.log(f"❌ 第 {i} 句生成失敗: {e}")
                    fail_count += 1
            
            # Enable play buttons (last generated audio)
            if generated_files:
                for btn in self.play_buttons.values():
                    btn.configure(state="normal", text="▶ 播放生成音訊")
            
            # Summary
            total_elapsed = f"{time.time() - batch_start:.2f}"
            self.logger.log(f"")
            self.logger.log(f"🏁 批次合成全部完成！成功: {success_count} 句 / 失敗: {fail_count} 句 / 總耗時: {total_elapsed} 秒")
            
            summary = f"批次合成完成！\n\n✅ 成功: {success_count} 句\n❌ 失敗: {fail_count} 句\n⏱️ 總耗時: {total_elapsed} 秒\n\n輸出資料夾:\n{self.output_dir_var.get()}"
            self.after(0, lambda: messagebox.showinfo("批次合成完成 🎉", summary))
            
        except Exception as e:
            self.logger.log(f"❌ 批次合成發生嚴重錯誤: {e}")
            import traceback
            tb_str = traceback.format_exc()
            self.logger.log(f"詳細追蹤:\n{tb_str}")
        finally:
            self.after(0, self._stop_progress_bar)

    def _infer_thread_run(self, model_dir, args, speed_rate):
        global VOXCPM_MODEL
        try:
            # Lazy load model if not loaded
            if VOXCPM_MODEL is None:
                self.logger.log("🔄 正在初次初始化並載入 VoxCPM2 模型... (模型大小為 2B，第一次載入可能需要 20-40 秒)")
                from voxcpm import VoxCPM
                VOXCPM_MODEL = VoxCPM.from_pretrained(model_dir, load_denoiser=False, optimize=not getattr(sys, 'frozen', False))
                self.logger.log("✅ 模型成功加載至記憶體/顯示記憶體！")

            self.logger.log("🚀 開始生成語音...")
            self.logger.log(f"參數設定: CFG Scale={args.get('cfg_value')}, 去噪步數={args.get('inference_timesteps')}, 語速={speed_rate}x, 標準化={args.get('normalize')}, 降噪={args.get('denoise')}")
            
            # Apply seed
            seed_val = args.pop("seed", None)
            if seed_val is not None:
                self.set_global_seed(seed_val)
                self.logger.log(f"🔒 使用固定種子 ID: {seed_val}")
            else:
                random_seed = random.randint(0, 10000000)
                self.set_global_seed(random_seed)
                self.logger.log(f"🎲 使用隨機種子 ID: {random_seed} (可複製此 ID 並勾選「固定隨機種子」來固定音色)")

            start_time = time.time()
            # Generate wav using VoxCPM
            wav = VOXCPM_MODEL.generate(**args)
            
            if speed_rate != 1.0:
                self.logger.log(f"⏳ 正在調整語速為 {speed_rate}x...")
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=speed_rate)
            
            # Save audio
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"voxcpm_{timestamp}.wav"
            filepath = os.path.join(self.output_dir_var.get(), filename)
            
            sf.write(filepath, wav, VOXCPM_MODEL.tts_model.sample_rate)
            
            self.logger.log(f"🎉 語音生成成功！耗時: {time.time() - start_time:.2f} 秒")
            self.logger.log(f"儲存路徑: {filepath}")
            
            # Save ref to last audio and enable ALL play buttons
            self.last_generated_audio = filepath
            for btn in self.play_buttons.values():
                btn.configure(state="normal", text="▶ 播放生成音訊")
            
            # Show success notification
            elapsed = f"{time.time() - start_time:.2f}"
            self.after(0, lambda: messagebox.showinfo("語音生成完成 🎉", f"語音合成成功！耗時: {elapsed} 秒\n\n儲存路徑:\n{filepath}"))
            
        except Exception as e:
            self.logger.log(f"❌ 生成失敗，錯誤資訊: {e}")
            import traceback
            tb_str = traceback.format_exc()
            self.logger.log(f"詳細追蹤:\n{tb_str}")
        finally:
            self.after(0, self._stop_progress_bar)

    def _stop_progress_bar(self):
        try:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)
        except Exception:
            pass

    def _toggle_seed_entry(self, func_type):
        if self.chk_seed_vars[func_type].get():
            self.entry_seeds[func_type].configure(state="normal")
        else:
            self.entry_seeds[func_type].configure(state="disabled")

    def _apply_voice_preset(self, lang, role):
        presets = {
            "tw_mandarin": {
                "臺灣老爺爺": "An elderly man, Taiwanese Mandarin accent, warm and gentle voice",
                "臺灣老奶奶": "An elderly woman, Taiwanese Mandarin accent, kind and soft voice",
                "臺灣年輕男生": "A young man, Taiwanese Mandarin accent, clear and energetic voice",
                "臺灣年輕女生": "A young woman, Taiwanese Mandarin accent, gentle and sweet voice",
                "臺灣小男孩": "A young boy, Taiwanese Mandarin accent, cute voice",
                "臺灣小女孩": "A young girl, Taiwanese Mandarin accent, lovely and sweet voice"
            },
            "japanese": {
                "老爺爺": "An old man, Japanese accent, warm and gentle voice",
                "老奶奶": "An old woman, Japanese accent, kind and soft voice",
                "年輕男生": "A young man, Japanese accent, clear and energetic voice",
                "年輕女生": "A young woman, Japanese accent, gentle and sweet voice",
                "小男孩": "A young boy, Japanese accent, cute voice",
                "小女孩": "A young girl, Japanese accent, lovely voice"
            },
            "us_english": {
                "老爺爺": "An old man, American accent, deep and warm voice",
                "老奶奶": "An old woman, American accent, kind and gentle voice",
                "年輕男生": "A young man, American accent, clear and energetic voice",
                "年輕女生": "A young woman, American accent, gentle and sweet voice",
                "小男孩": "A young boy, American accent, cute voice",
                "小女孩": "A young girl, American accent, lovely voice"
            },
            "korean": {
                "老爺爺": "An old man, Korean accent, warm and gentle voice",
                "老奶奶": "An old woman, Korean accent, kind and soft voice",
                "年輕男生": "A young man, Korean accent, clear and energetic voice",
                "年輕女生": "A young woman, Korean accent, gentle and sweet voice",
                "小男孩": "A young boy, Korean accent, cute voice",
                "小女孩": "A young girl, Korean accent, lovely voice"
            }
        }
        
        desc = presets.get(lang, {}).get(role, "")
        if not desc:
            return
            
        sentences = {
            "tw_mandarin": "哈囉！這是使用語音設計無中生有出來的聲音。",
            "japanese": "こんにちは！これは音声設計で新しく生成された声です。",
            "us_english": "Hello! This is a voice generated from scratch using voice design.",
            "korean": "안녕하세요! 이것은 음성 디자인으로 새로 생성된 목소리입니다."
        }
        
        sentence = sentences.get(lang, "")
        new_line = f"({desc}) {sentence}"
        
        current_content = self.txt_design.get("0.0", "end-1c")
        
        if not current_content.strip():
            # If empty or only whitespace, set the content directly
            self.txt_design.delete("0.0", "end")
            self.txt_design.insert("0.0", new_line)
        else:
            # Append as a new line
            if current_content.endswith("\n"):
                self.txt_design.insert("end", new_line)
            else:
                self.txt_design.insert("end", "\n" + new_line)
                
        # Reset all option menus back to their default titles
        default_titles = {
            "tw_mandarin": "臺灣國語 (選擇音色)",
            "japanese": "日本語 (選擇音色)",
            "us_english": "美國英語 (選擇音色)",
            "korean": "韓語 (選擇音色)"
        }
        
        for k, menu in self.preset_menus.items():
            menu.set(default_titles[k])

    def set_global_seed(self, seed):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Playback functions
    def toggle_playback(self):
        global PLAYBACK_THREAD, PLAYBACK_STOP_EVENT, IS_PLAYING
        
        if not self.last_generated_audio or not os.path.exists(self.last_generated_audio):
            return

        # Stop recorded playback if active
        if hasattr(self, 'is_playing_recorded') and self.is_playing_recorded:
            self.recorded_play_stop_event.set()
            sd.stop()
            self.btn_play_recorded.configure(text="▶ 播放錄音")
            self.is_playing_recorded = False
            time.sleep(0.1)

        if IS_PLAYING:
            # Stop playback
            PLAYBACK_STOP_EVENT.set()
            sd.stop()
            for btn in self.play_buttons.values():
                btn.configure(text="▶ 播放生成音訊")
            IS_PLAYING = False
        else:
            # Start playback
            PLAYBACK_STOP_EVENT.clear()
            for btn in self.play_buttons.values():
                btn.configure(text="⏹ 停止播放")
            
            def _on_finished():
                for btn in self.play_buttons.values():
                    btn.configure(text="▶ 播放生成音訊")
                
            PLAYBACK_THREAD = threading.Thread(
                target=SoundPlayer.play_audio, 
                args=(self.last_generated_audio, PLAYBACK_STOP_EVENT, _on_finished),
                daemon=True
            )
            PLAYBACK_THREAD.start()

    def toggle_recorded_playback(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            return
            
        # Stop any generated audio playback if it is running
        global IS_PLAYING, PLAYBACK_STOP_EVENT
        if IS_PLAYING:
            PLAYBACK_STOP_EVENT.set()
            sd.stop()
            for btn in self.play_buttons.values():
                btn.configure(text="▶ 播放生成音訊")
            IS_PLAYING = False
            time.sleep(0.1)
            
        if self.is_playing_recorded:
            self.recorded_play_stop_event.set()
            sd.stop()
            self.btn_play_recorded.configure(text="▶ 播放錄音")
            self.is_playing_recorded = False
        else:
            self.recorded_play_stop_event.clear()
            self.btn_play_recorded.configure(text="⏹ 停止播放")
            self.is_playing_recorded = True
            
            def _on_finished():
                self.btn_play_recorded.configure(text="▶ 播放錄音")
                self.is_playing_recorded = False
                
            self.recorded_play_thread = threading.Thread(
                target=SoundPlayer.play_audio,
                args=(self.recorded_filepath, self.recorded_play_stop_event, _on_finished),
                daemon=True
            )
            self.recorded_play_thread.start()

    def populate_recording_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []
            default_device_idx = sd.default.device[0] # Default input device idx
            default_device_str = "預設輸入裝置"
            
            for idx, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    device_name = f"{idx}: {d['name']}"
                    input_devices.append(device_name)
                    if idx == default_device_idx:
                        default_device_str = device_name
            
            if not input_devices:
                self.menu_devices.configure(values=["無可用麥克風"])
                self.recording_device_var.set("無可用麥克風")
                self.btn_record.configure(state="disabled")
                self.logger.log("⚠️ 系統未偵測到任何音訊輸入裝置（麥克風）。")
            else:
                self.menu_devices.configure(values=input_devices)
                # Find if default exists in list, else use the first one
                if default_device_str in input_devices:
                    self.recording_device_var.set(default_device_str)
                else:
                    self.recording_device_var.set(input_devices[0])
                self.btn_record.configure(state="normal")
                self.logger.log(f"🎙️ 已載入 {len(input_devices)} 個音訊輸入裝置，預設為: {self.recording_device_var.get()}")
        except Exception as e:
            self.logger.log(f"⚠️ 取得音訊設備失敗: {e}")
            self.menu_devices.configure(values=["設備取得出錯"])
            self.recording_device_var.set("設備取得出錯")
            self.btn_record.configure(state="disabled")

    def toggle_recording(self):
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.btn_record.configure(text="🔴 開始錄音", fg_color="#F44336", hover_color="#D32F2F")
            
            if self.recording_stream:
                try:
                    self.recording_stream.stop()
                    self.recording_stream.close()
                except Exception:
                    pass
                self.recording_stream = None
            
            # Reset visual waveform to a flat line when stopping
            try:
                self.wave_canvas.delete("wave")
                self.wave_canvas.create_line(0, 22, 250, 22, fill="#555555", tags="wave", width=1)
            except Exception:
                pass

            # Save the recorded file
            if self.recorded_frames:
                try:
                    audio_data = np.concatenate(self.recorded_frames, axis=0)
                    out_dir = self.output_dir_var.get()
                    os.makedirs(out_dir, exist_ok=True)
                    self.recorded_filepath = os.path.join(out_dir, "user_record_test.wav")
                    
                    sf.write(self.recorded_filepath, audio_data, 44100)
                    
                    duration = len(audio_data) / 44100.0
                    self.lbl_record_status.configure(
                        text=f"✅ 錄音完成！共 {duration:.1f} 秒 (已存至 outputs/user_record_test.wav)",
                        text_color="#4CAF50"
                    )
                    self.logger.log(f"🎤 錄音成功儲存: {self.recorded_filepath}，時長: {duration:.1f} 秒")
                    
                    # Enable playback and apply buttons
                    self.btn_play_recorded.configure(state="normal")
                    self.btn_apply_clone.configure(state="normal")
                    self.btn_apply_ultimate.configure(state="normal")
                except Exception as e:
                    self.lbl_record_status.configure(text=f"❌ 儲存音檔失敗: {e}", text_color="#F44336")
                    self.logger.log(f"❌ 儲存錄音時發生錯誤: {e}")
            else:
                self.lbl_record_status.configure(text="⚠️ 未錄製到任何音訊資料。", text_color="#FF9800")
        else:
            # Start recording
            device_str = self.recording_device_var.get()
            if "無可用麥克風" in device_str or "設備取得出錯" in device_str:
                messagebox.showerror("錯誤", "無可用的錄音輸入裝置！")
                return
            
            # Parse device index
            try:
                device_idx = int(device_str.split(":")[0])
            except Exception:
                device_idx = None # fallback to default
                
            self.recorded_frames = []
            self.vis_buffer = np.zeros(4000) # Reset visualization buffer
            self.is_recording = True
            self.btn_record.configure(text="⏹ 停止錄音", fg_color="#607D8B", hover_color="#455A64")
            self.lbl_record_status.configure(text="🎙️ 正在錄音中...", text_color="#E91E63")
            
            # Disable other actions while recording
            self.btn_play_recorded.configure(state="disabled")
            self.btn_apply_clone.configure(state="disabled")
            self.btn_apply_ultimate.configure(state="disabled")
            
            # Helper callback
            def callback(indata, frames, time_info, status):
                if self.is_recording:
                    self.recorded_frames.append(indata.copy())
                    # Shift and append to visualization buffer
                    self.vis_buffer = np.roll(self.vis_buffer, -frames)
                    self.vis_buffer[-frames:] = indata[:, 0]
            
            try:
                self.recording_stream = sd.InputStream(
                    samplerate=44100,
                    channels=1,
                    device=device_idx,
                    callback=callback
                )
                self.recording_stream.start()
                
                # Start UI timer update
                start_time = time.time()
                self._update_recording_timer(start_time)
                self.logger.log("🎤 開始現場錄音，請對著麥克風朗讀文本...")
            except Exception as e:
                self.is_recording = False
                self.btn_record.configure(text="🔴 開始錄音", fg_color="#F44336", hover_color="#D32F2F")
                self.lbl_record_status.configure(text=f"❌ 開啟錄音設備失敗: {e}", text_color="#F44336")
                self.logger.log(f"❌ 無法開啟錄音裝置: {e}")

    def _update_recording_timer(self, start_time):
        if self.is_recording:
            elapsed = time.time() - start_time
            self.lbl_record_status.configure(text=f"🎙️ 正在錄音中 ({elapsed:.1f} 秒)...")
            
            # Draw real-time voice ripple/waveform
            try:
                width = 250
                height = 45
                mid_y = height / 2
                points = []
                step = 4000 // width
                
                for i in range(width):
                    val = self.vis_buffer[i * step]
                    # Apply gain to make it visually obvious
                    y = mid_y - (val * mid_y * 2.5)
                    y = max(2, min(height - 2, y))
                    points.append((i, y))
                    
                self.wave_canvas.delete("wave")
                for idx in range(len(points) - 1):
                    self.wave_canvas.create_line(
                        points[idx][0], points[idx][1],
                        points[idx+1][0], points[idx+1][1],
                        fill="#E91E63", tags="wave", width=2
                    )
            except Exception:
                pass
            
            # Stop automatically after 15 seconds to prevent extremely long audio
            if elapsed >= 15.0:
                self.logger.log("⏱️ 已達錄音上限 15 秒，系統自動停止錄音。")
                self.toggle_recording()
            else:
                self.after(50, lambda: self._update_recording_timer(start_time)) # Update every 50ms for smoother animation

    def use_recorded_for_clone(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            messagebox.showwarning("警告", "請先完成錄音！")
            return
        self.entry_clone_ref.delete(0, tk.END)
        self.entry_clone_ref.insert(0, self.recorded_filepath)
        self.select_view("clone")
        self.logger.log(f"🎤 已將現場錄音套用至「聲音複製」的參考語音檔。")

    def use_recorded_for_ultimate(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            messagebox.showwarning("警告", "請先完成錄音！")
            return
        self.entry_ult_ref.delete(0, tk.END)
        self.entry_ult_ref.insert(0, self.recorded_filepath)
        self.entry_ult_prompt.delete(0, tk.END)
        self.entry_ult_prompt.insert(0, self.record_prompt_text)
        self.select_view("ultimate")
        self.logger.log(f"🎤 已將現場錄音與對應逐字稿套用至「極限複製」。")

    def open_output_folder(self):
        out_dir = self.output_dir_var.get()
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            os.makedirs(out_dir, exist_ok=True)
            os.startfile(out_dir)

    # Downloader thread
    def start_model_download(self):
        if self.is_downloading:
            self.download_cancelled_event.set()
            self.btn_download.configure(text="正在停止...", state="disabled")
            self.logger.log("🛑 正在要求中斷下載程序，請稍候...")
            return

        model_dir = self.model_dir_var.get()
        os.makedirs(model_dir, exist_ok=True)
        
        # Check which files are missing
        REQUIRED_MODEL_FILES = {
            "config.json": 4000,
            "special_tokens_map.json": 1500,
            "tokenization_voxcpm2.py": 2500,
            "tokenizer.json": 3500000,
            "tokenizer_config.json": 4500,
            "audiovae.pth": 350000000,
            "model.safetensors": 4000000000
        }
        
        FILE_DISPLAY_SIZES = {
            "config.json": "4.2 KB",
            "special_tokens_map.json": "1.6 KB",
            "tokenization_voxcpm2.py": "2.8 KB",
            "tokenizer.json": "3.5 MB",
            "tokenizer_config.json": "4.9 KB",
            "audiovae.pth": "359.5 MB",
            "model.safetensors": "4.27 GB"
        }
        
        missing_files = []
        for filename, min_size in REQUIRED_MODEL_FILES.items():
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) < min_size:
                missing_files.append(filename)
                
        if not missing_files:
            self.logger.log("✅ 檢查完成：本機已存在所有完整模型檔案，無須下載！")
            messagebox.showinfo("檢查完成", "本機已存在所有完整模型檔案，無須重新下載！")
            return
            
        self.logger.log("🔍 開始檢查本機模型檔案...")
        self.logger.log("本機尚缺少下列模型檔案，將進行下載：")
        for f in missing_files:
            self.logger.log(f"  - {f} ({FILE_DISPLAY_SIZES[f]})")
            
        prompt_msg = "偵測到本機缺少下列模型檔案，即將開始下載：\n" + "\n".join([f"• {f} ({FILE_DISPLAY_SIZES[f]})" for f in missing_files]) + "\n\n是否確認開始下載？"
        if not messagebox.askyesno("確認下載", prompt_msg):
            self.logger.log("❌ 使用者已取消下載流程。")
            return
            
        # Set state to downloading
        self.is_downloading = True
        self.download_cancelled_event.clear()
        
        self.btn_download.configure(text="停止下載", fg_color="#F44336", hover_color="#D32F2F")
        self.lbl_dl_status.configure(text="準備下載中...")
        
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0.0)
        
        source = self.downloader_source_var.get()
        threading.Thread(target=self._download_thread_run, args=(source, model_dir, missing_files), daemon=True).start()

    def _download_thread_run(self, source, model_dir, files_to_download):
        if "ModelScope" in source:
            base_url = "https://modelscope.cn/api/v1/models/OpenBMB/VoxCPM2/repo?Revision=master&FilePath="
            self.logger.log("🔄 開始下載缺失的模型檔案 (來源: ModelScope)...")
        elif "鏡像" in source:
            base_url = "https://hf-mirror.com/openbmb/VoxCPM2/resolve/main/"
            self.logger.log("🔄 開始下載缺失的模型檔案 (來源: Hugging Face 鏡像)...")
        else:
            base_url = "https://huggingface.co/openbmb/VoxCPM2/resolve/main/"
            self.logger.log("🔄 開始下載缺失的模型檔案 (來源: Hugging Face 官方)...")
            
        self.logger.log(f"待下載清單: {', '.join(files_to_download)}")
        
        success_all = True
        cancelled = False
        
        try:
            for idx, filename in enumerate(files_to_download, 1):
                if self.download_cancelled_event.is_set():
                    cancelled = True
                    break
                    
                url = base_url + filename
                filepath = os.path.join(model_dir, filename)
                temp_filepath = filepath + ".downloading"
                
                self.logger.log(f"📥 正在下載檔案 [{idx}/{len(files_to_download)}]: {filename}...")
                self._safe_update_ui(f"正在下載 [{idx}/{len(files_to_download)}]: {filename}...", 0.0)
                
                # Run download
                success = self._download_single_file(url, temp_filepath, filename, idx, len(files_to_download))
                
                if self.download_cancelled_event.is_set():
                    cancelled = True
                    if os.path.exists(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                    break
                    
                if not success:
                    success_all = False
                    self.logger.log(f"❌ {filename} 下載失敗。")
                    if os.path.exists(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                    break
                else:
                    # Rename temp file to final filepath
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
                    os.rename(temp_filepath, filepath)
                    self.logger.log(f"✅ {filename} 下載完成！")
                    
            if cancelled:
                self.logger.log("🛑 下載程序已成功中斷。")
                self._safe_update_ui("下載已中斷", 0.0)
                self.after(0, lambda: messagebox.showwarning("下載中斷", "模型下載程序已成功中斷！"))
            elif not success_all:
                self.logger.log("❌ 模型下載失敗。請檢查網路或更換下載伺服器來源。")
                self._safe_update_ui("下載失敗", 0.0)
                self.after(0, lambda: messagebox.showerror("下載失敗", "下載模型時發生錯誤，請檢查下方的日誌資訊。"))
            else:
                self.logger.log("🎉 所有缺失檔案下載與部署完成！您現在可以開始使用模型。")
                self._safe_update_ui("下載完成", 1.0)
                self.after(0, lambda: messagebox.showinfo("下載成功", f"所有缺失模型檔案已成功部署至:\n{model_dir}"))
                
        except Exception as e:
            self.logger.log(f"❌ 下載程序發生嚴重異常: {e}")
            self._safe_update_ui("發生錯誤", 0.0)
            self.after(0, lambda: messagebox.showerror("錯誤", f"下載程序發生異常:\n{e}"))
        finally:
            self.is_downloading = False
            self.btn_download.configure(
                text="開始下載 / 檢查模型", 
                fg_color="#9C27B0", 
                hover_color="#7B1FA2", 
                state="normal"
            )
            self.after(0, self._stop_progress_bar)

    def _download_single_file(self, url, temp_filepath, filename, idx, total_count):
        import urllib.request
        import urllib.error
        import time
        import socket
        
        max_retries = 5
        retry_delay = 3
        timeout = 20  # 20 seconds timeout for connect and read
        
        for attempt in range(1, max_retries + 1):
            if self.download_cancelled_event.is_set():
                return False
                
            bytes_so_far = 0
            if os.path.exists(temp_filepath):
                bytes_so_far = os.path.getsize(temp_filepath)
                
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            if bytes_so_far > 0:
                req.add_header('Range', f'bytes={bytes_so_far}-')
                
            os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
            
            try:
                if attempt > 1:
                    self.logger.log(f"正在連線到下載伺服器 (嘗試 {attempt}/{max_retries})...")
                
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status_code = response.getcode()
                    
                    if status_code == 206:
                        # Resuming download
                        file_mode = 'ab'
                        content_range = response.info().get('Content-Range', '')
                        total_size = 0
                        if '/' in content_range:
                            try:
                                total_size = int(content_range.split('/')[-1])
                            except ValueError:
                                pass
                        if not total_size:
                            total_size = bytes_so_far + int(response.info().get('Content-Length', 0))
                        self.logger.log(f"➡️ 偵測到未完成下載，已從 {bytes_so_far / (1024*1024):.1f}MB 處繼續下載...")
                    else:
                        # Normal download from scratch (or server does not support partial content)
                        bytes_so_far = 0
                        file_mode = 'wb'
                        total_size = int(response.info().get('Content-Length', 0))
                        
                    block_size = 256 * 1024  # 256 KB chunks
                    start_time = time.time()
                    last_ui_update = 0
                    session_bytes_downloaded = 0
                    
                    with open(temp_filepath, file_mode) as f:
                        while True:
                            if self.download_cancelled_event.is_set():
                                return False
                                
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                                
                            f.write(chunk)
                            bytes_so_far += len(chunk)
                            session_bytes_downloaded += len(chunk)
                            
                            current_time = time.time()
                            if current_time - last_ui_update > 0.15:
                                last_ui_update = current_time
                                elapsed = current_time - start_time
                                speed = (session_bytes_downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                                
                                if total_size > 0:
                                    percent = float(bytes_so_far) / total_size * 100
                                    remaining_bytes = total_size - bytes_so_far
                                    speed_bytes = session_bytes_downloaded / elapsed if elapsed > 0 else 0
                                    remaining_seconds = remaining_bytes / speed_bytes if speed_bytes > 0 else 0
                                    
                                    rem_min, rem_sec = divmod(int(remaining_seconds), 60)
                                    rem_str = f"{rem_min:02d}:{rem_sec:02d}"
                                    
                                    status_text = f"正在下載 [{idx}/{total_count}] {filename}: {percent:.1f}% | {bytes_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB | {speed:.2f} MB/s | 剩餘時間: {rem_str}"
                                    progress_val = bytes_so_far / total_size
                                else:
                                    status_text = f"正在下載 [{idx}/{total_count}] {filename}: {bytes_so_far / (1024*1024):.1f}MB | {speed:.2f} MB/s"
                                    progress_val = 0.5
                                    
                                self._safe_update_ui(status_text, progress_val)
                                
                # If we successfully completed the loop and written everything
                return True
                
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    self.logger.log(f"⚠️ 伺服器拒絕續傳 (HTTP 416)，將刪除暫存檔並重頭下載...")
                    if os.path.exists(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                else:
                    self.logger.log(f"⚠️ 連線發生 HTTP 錯誤 (代碼 {e.code}): {e.reason}")
                
            except (urllib.error.URLError, socket.timeout, ConnectionResetError) as e:
                self.logger.log(f"⚠️ 連線超時或異常: {e}")
                
            except Exception as e:
                self.logger.log(f"⚠️ 下載時發生非預期錯誤: {e}")
                
            # Wait before retry
            if attempt < max_retries:
                self.logger.log(f"⏱️ 將於 {retry_delay} 秒後進行第 {attempt + 1} 次重試...")
                for _ in range(retry_delay * 10):
                    if self.download_cancelled_event.is_set():
                        return False
                    time.sleep(0.1)
                    
        return False

    def _safe_update_ui(self, text, val):
        self.after(0, lambda: self._update_ui_elements(text, val))
        
    def _update_ui_elements(self, text, val):
        try:
            self.lbl_dl_status.configure(text=text)
            self.progress_bar.set(val)
        except Exception:
            pass

if __name__ == "__main__":
    # Add exception logger helper
    import sys
    app = App()
    app.mainloop()
