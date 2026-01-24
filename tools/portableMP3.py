import os
import argparse
import subprocess
from pathlib import Path

def run_ffmpeg(input_file, output_file, bitrate, include_cover, start=None, duration=None):
    """Core Engine: Optimized for UltraSinger 3.10 Sweet Spot."""
    clean_name = Path(input_file).stem 

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', 
        '-i', str(input_file)
    ]
    
    if start: 
        cmd.insert(cmd.index('-i'), '-ss')
        cmd.insert(cmd.index('-ss')+1, start)
    if duration: 
        cmd.extend(['-t', str(duration)])
        
    # Audio Mapping
    cmd.extend(['-map', '0:a'])

    # Cover Art Logic
    if include_cover:
        cmd.extend(['-map', '0:v?', '-c:v', 'copy']) # Copy video/image stream if exists
    
    cmd.extend([
        '-map_metadata', '0',       # Copy original tags
        '-id3v2_version', '3',      # UltraSinger 3.10 sweet spot
        '-metadata', f'title={clean_name}', 
        '-b:a', bitrate,            # Flexible Bitrate (128k or Common)
        '-ar', '44100',             
        str(output_file)
    ])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraSinger Multi-Tool Optimizer")
    parser.add_argument("-i", "--input", required=True, help="Input File or Directory Tree Root")
    parser.add_argument("-o", "--output", required=True, help="Target Output Tree (Gets rid of hard links)")
    parser.add_argument("-b", "--bitrate", default="128k", choices=["128k", "192k", "256k", "320k"], help="Bitrate (Default: 128k)")
    parser.add_argument("--cover", action="store_true", help="Include Cover Art (Default: Stripped)")
    parser.add_argument("--start", help="Start time (HH:MM:SS)")
    parser.add_argument("--duration", help="Length in seconds")
    
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    # Determine if we are processing a single file or a tree
    files_to_process = []
    if input_root.is_file():
        files_to_process.append(input_root)
    else:
        # Recursive walk to find audio files
        extensions = ('.mp3', '.wav', '.flac', '.m4a')
        files_to_process = [f for f in input_root.rglob('*') if f.suffix.lower() in extensions]

    print(f"🚀 Processing {len(files_to_process)} tracks into {output_root}...")

    for f in files_to_process:
        # Maintain Tree Structure: Calculate relative path from input root
        relative_path = f.relative_to(input_root if input_root.is_dir() else input_root.parent)
        target_path = output_root / relative_path.with_suffix('.mp3')
        
        # Create the sub-directories in the new tree
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"   Converting: {relative_path}")
        success = run_ffmpeg(f, target_path, args.bitrate, args.cover, args.start, args.duration)
        
        if not success:
            print(f"❌ Failed: {f.name}")

    print(f"\n✅ Tree Migration Complete. Files located in: {output_root}")
