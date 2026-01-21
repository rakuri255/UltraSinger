#!/usr/bin/env python3
import os
import subprocess
import glob
import sys
import readline
import platform
import shutil

# Colors
CYAN = '\033[0;36m'
GRN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m'

# Setup readline for better input editing
readline.parse_and_bind('set editing-mode emacs')

def get_safe_executable(name):
    """Finds the absolute path of an executable to prevent path hijacking."""
    exe_path = shutil.which(name)
    if not exe_path:
        print(f"{RED}Error: {name} is not installed or not in PATH.{NC}")
        sys.exit(1)
    return exe_path

def check_dependencies():
    """Ensure ffmpeg is available and return its absolute path."""
    return get_safe_executable('ffmpeg')

def get_audio_files():
    """Scan directory for common audio formats."""
    extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a', '*.ogg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(ext))
    return sorted(files)

def clean_path(path):
    """Remove quotes from drag-and-dropped paths."""
    return path.strip().replace("'", "").replace('"', "")

def play_file(filepath):
    """Opens the file in the default system player using absolute paths."""
    if not os.path.exists(filepath):
        print(f"{RED}File not found: {filepath}{NC}")
        return

    print(f"{CYAN}🎧 Playing: {filepath}{NC}")
    current_os = platform.system()
    abs_filepath = os.path.abspath(filepath)
    
    try:
        if current_os == 'Windows':
            # os.startfile is secure as it doesn't invoke a shell
            os.startfile(abs_filepath)
        elif current_os == 'Darwin':  # macOS
            cmd = get_safe_executable('open')
            subprocess.run([cmd, abs_filepath], check=True)
        else:  # Linux
            cmd = get_safe_executable('xdg-open')
            subprocess.run([cmd, abs_filepath], check=True)
    except Exception as e:
        print(f"{RED}Could not open player: {e}{NC}")

def select_input(prompt_text, file_list):
    """Helper to handle selecting a file by number or path."""
    while True:
        raw = input(f"{GRN}⌨️  {prompt_text} (Number or Path):{NC} ").strip()
        if not raw:
            continue
            
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(file_list):
                return file_list[idx]
            else:
                print(f"{RED}Invalid number.{NC}")
                continue
        
        clean = clean_path(raw)
        if os.path.exists(clean):
            return clean
        else:
            print(f"{RED}File path not found.{NC}")

def main():
    ffmpeg_bin = check_dependencies()
    print(f"{CYAN}--------------------------------------------------------")
    print("🎚️  ULTRASINGER MIXING DESK & PREVIEW")
    print(f"--------------------------------------------------------{NC}")

    files = get_audio_files()
    if files:
        print(f"{CYAN}Found audio files:{NC}")
        for i, file in enumerate(files):
            print(f"{i+1}) {file}")
        print("--------------------------------------------------------")
    else:
        print(f"{RED}No audio files found in current directory.{NC}")

    vocals = select_input("Select VOCAL", files)
    instrumental = select_input("Select INSTRUMENTAL", files)

    while True:
        print(f"\n{CYAN}Select Action:{NC}")
        print(f"1) {GRN}Karaoke{NC}       (Vocals 1.2x | Inst 1.2x)")
        print(f"2) {GRN}Instrumental{NC}  (Vocals 0.0x | Inst 1.2x)")
        print(f"3) {GRN}Guide Track{NC}   (Vocals 0.3x | Inst 1.2x)")
        print(f"4) {CYAN}Preview Vocals{NC}")
        print(f"5) {CYAN}Preview Instrumental{NC}")
        print(f"6) {RED}Exit{NC}")
        
        choice = input(">> Choice [1-6]: ").strip()

        match choice:
            case "4":
                play_file(vocals)
                continue
            case "5":
                play_file(instrumental)
                continue
            case "6":
                print("Bye!")
                sys.exit(0)
            case "1":
                config = {"name": "song.mp3", "v": "1.2", "i": "1.2"}
            case "2":
                config = {"name": "instrumental.mp3", "v": "0.0", "i": "1.2"}
            case "3":
                config = {"name": "guide.mp3", "v": "0.3", "i": "1.2"}
            case _:
                print(f"{RED}Invalid selection.{NC}")
                continue

        out_file = input(f"{GRN}Filename (Enter for '{config['name']}'):{NC} ") or config['name']

        filter_complex = (
            f"[0:a]volume={config['v']}[v];"
            f"[1:a]volume={config['i']}[i];"
            f"[v][i]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        
        # Build command as a list to avoid shell=True
        cmd_args = [
            ffmpeg_bin, "-y",
            "-i", vocals,
            "-i", instrumental,
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-b:a", "320k",
            out_file
        ]

        mixing = True
        while mixing:
            print(f"\n{CYAN}-----------------------------------------------------------")
            print(f"🛠️  PROPOSED COMMAND:")
            print(f"{NC}{' '.join(cmd_args)}")
            print(f"-----------------------------------------------------------")
            
            user_choice = input(f"Type {GRN}'run'{NC}, {CYAN}'play'{NC} (result), {CYAN}'back'{NC} or {RED}'exit'{NC}: ").lower().strip()

            if user_choice == 'run':
                try:
                    print(f"{CYAN}Rendering...{NC}")
                    # shell=False is used by default when passing a list
                    subprocess.run(cmd_args, check=True)
                    print(f"{GRN}✅ Success! Saved to: ./{out_file}{NC}")
                except subprocess.CalledProcessError:
                    print(f"{RED}❌ FFmpeg failed.{NC}")
            elif user_choice == 'play':
                play_file(out_file)
            elif user_choice == 'back':
                mixing = False 
            elif user_choice == 'exit':
                sys.exit(0)

if __name__ == "__main__":
    main()
