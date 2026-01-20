#!/usr/bin/env python3
import os
import subprocess
import glob
import sys
import readline
import platform

# Colors
CYAN = '\033[0;36m'
GRN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m'

# Setup readline for better input editing
readline.parse_and_bind('set editing-mode emacs')

def check_dependencies():
    """Ensure ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"{RED}Error: ffmpeg is not installed or not in PATH.{NC}")
        sys.exit(1)

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
    """Opens the file in the default system player (Cross-platform)."""
    if not os.path.exists(filepath):
        print(f"{RED}File not found: {filepath}{NC}")
        return

    print(f"{CYAN}🎧 Playing: {filepath}{NC}")
    current_os = platform.system()
    
    try:
        if current_os == 'Windows':
            os.startfile(filepath)
        elif current_os == 'Darwin':  # macOS
            subprocess.run(['open', filepath], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', filepath], check=True)
    except Exception as e:
        print(f"{RED}Could not open player: {e}{NC}")

def select_input(prompt_text, file_list):
    """Helper to handle selecting a file by number or path."""
    while True:
        raw = input(f"{GRN}⌨️  {prompt_text} (Number or Path):{NC} ").strip()
        if not raw:
            continue
            
        # If user entered a number
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(file_list):
                return file_list[idx]
            else:
                print(f"{RED}Invalid number.{NC}")
                continue
        
        # If user entered a path
        clean = clean_path(raw)
        if os.path.exists(clean):
            return clean
        else:
            print(f"{RED}File path not found.{NC}")

def main():
    check_dependencies()
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

    # 1. SELECT FILES
    # We allow manual path entry even if no files were globbed
    vocals = select_input("Select VOCAL", files)
    instrumental = select_input("Select INSTRUMENTAL", files)

    # 2. ACTION MENU LOOP
    while True:
        print(f"\n{CYAN}Select Action:{NC}")
        print(f"1) {GRN}Karaoke{NC}       (Vocals 1.2x | Inst 1.2x)")
        print(f"2) {GRN}Instrumental{NC}  (Vocals 0.0x | Inst 1.2x)")
        print(f"3) {GRN}Guide Track{NC}   (Vocals 0.3x | Inst 1.2x)")
        print(f"4) {CYAN}Preview Vocals{NC}")
        print(f"5) {CYAN}Preview Instrumental{NC}")
        print(f"6) {RED}Exit{NC}")
        
        choice = input(">> Choice [1-6]: ").strip()

        # Mixing Profiles
        # Adjusted default volume to 1.2 based on feedback
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

        # 3. BUILD FFmpeg COMMAND
        out_file = input(f"{GRN}Filename (Enter for '{config['name']}'):{NC} ") or config['name']

        # Using normalized=0 prevents amix from dropping volume automatically,
        # so our 1.2 volume filter effectively boosts gain.
        filter_complex = (
            f"[0:a]volume={config['v']}[v];"
            f"[1:a]volume={config['i']}[i];"
            f"[v][i]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        
        current_cmd = (
            f'ffmpeg -y -i "{vocals}" -i "{instrumental}" '
            f'-filter_complex "{filter_complex}" -map "[aout]" '
            f'-b:a 320k "{out_file}"'
        )

        # 4. SURGERY LOOP
        # Allows editing the command before running it
        mixing = True
        while mixing:
            print(f"\n{CYAN}-----------------------------------------------------------")
            print(f"🛠️  CURRENT COMMAND:")
            print(f"{NC}{current_cmd}")
            print(f"-----------------------------------------------------------")
            
            user_choice = input(f"Type {GRN}'run'{NC}, {GRN}'edit'{NC}, {CYAN}'play'{NC} (result), {CYAN}'back'{NC}  {RED}'exit'{NC}: ").lower().strip()

            if user_choice == 'run':
                try:
                    print(f"{CYAN}Rendering...{NC}")
                    subprocess.run(current_cmd, shell=True, check=True)
                    print(f"{GRN}✅ Success! Saved to: ./{out_file}{NC}")
                except subprocess.CalledProcessError:
                    print(f"{RED}❌ FFmpeg failed.{NC}")
            elif user_choice == 'edit':
                # Pre-fill input buffer with current command for easy editing
                readline.add_history(current_cmd)
                print(f"{CYAN}Edit command (use arrow keys):{NC}")
                new_cmd = input("> ")
                if new_cmd.strip():
                    current_cmd = new_cmd
            elif user_choice == 'play':
                play_file(out_file)
            elif user_choice == 'back':
                mixing = False # Breaks inner loop, returns to menu
            elif user_choice == 'exit':
                sys.exit(0)

if __name__ == "__main__":
    main()
