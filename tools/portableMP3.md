# UltraSinger Portable MP3 Optimizer

A high-performance Python utility designed to compress and migrate large music archives for mobile devices, flip phones, and portable storage. Optimized specifically for the **UltraSinger 3.10** ecosystem.

## 📖 Description

This tool streamlines the process of converting high-bitrate or lossless music archives (FLAC, WAV, M4A) into mobile-friendly 128kbps MP3s. It features intelligent deduplication to clean up "dirty" archives and ensures metadata compatibility with legacy hardware.

## 🚩 Features

* **Bitrate Optimization**: Defaults to 128kbps (the "sweet spot" for storage vs. quality on mobile).
* **Intelligent Deduplication**: Automatically detects if files in the root directory match files in subdirectories; prioritizes organized album paths over loose root files.
* **Broad Compatibility**: Uses `ID3v2.3` and `44.1kHz` sampling to ensure album art and audio play correctly on basic handsets (e.g., Coosea SL006D).
* **Flexible Structure**: Supports both mirrored directory trees and flattened outputs.

## 🛠️ Usage

```bash
python3 portableMP3.py -i <input_folder> -o <output_folder> [FLAGS]

## 🛠️ Flag Documentation & Logic

### `--flat` (Flatten Output)
The `--flat` flag modifies the destination path logic. Instead of mirroring the source directory tree, it strips all path information and places every converted file directly into the root of the target folder.

**Use Case:** * Essential for MTP transfers to devices with slow file indexing.
* Required for legacy players/car stereos that do not support nested folders.

### 🧠 Intelligent Deduplication
The script now includes a "Path-Depth Priority" check to handle redundant files common in large archives (e.g., songs appearing in both the root and an album subfolder).

**How it works:**
1. Scans the input tree for all compatible audio files.
2. If two files share the same filename, the script calculates the path depth.
3. The file located **deeper** in the directory tree (e.g., inside an Album folder) is prioritized.
4. The redundant "root" version is skipped to save storage and prevent duplicate entries in mobile playlists.



---

## 🎨 Metadata & Album Art 
The `--cover` flag has been specifically tuned for hardware compatibility.

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **ID3 Version** | `v2.3` | Fixes "Missing Art" issues on flip phones (SL006D). |
| **Sample Rate** | `44100 Hz` | Standard for internal DACs on mobile devices. |
| **Mapping** | `0:v?` | Non-blocking; only attempts to copy art if the stream exists. |



---

## 🚀 Execution Examples

**Preserve Tree (Default):**
```bash
python3 portableMP3.py -i "~/Downloads/PonyArchive" -o "/media/nvme/AI_Work/Inputs" --cover
