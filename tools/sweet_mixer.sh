#!/bin/bash
# UltraSinger Audio Profile Mixer

# Colors
CYAN='\033[0;36m'
GRN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}--------------------------------------------------------"
echo -e "🎚️  ULTRASINGER MIXING DESK"
echo -e "--------------------------------------------------------${NC}"

# 1. SCAN FOR LOCAL FILES
shopt -s nullglob
FILES=( *.{wav,flac,mp3} )

if [ ${#FILES[@]} -gt 0 ]; then
    echo -e "${CYAN}Found audio files in this folder:${NC}"
    for i in "${!FILES[@]}"; do
        echo -e "$((i+1))) ${FILES[$i]}"
    done
    echo "--------------------------------------------------------"
fi

# 2. SELECT VOCALS
echo -e "${GRN}⌨️  Select VOCAL (Number or Path):${NC}"
read -e -r v_raw
if [[ "$v_raw" =~ ^[0-9]+$ ]] && [ "$v_raw" -le "${#FILES[@]}" ]; then
    VOCALS="${FILES[$((v_raw-1))]}"
else
    v_clean="${v_raw//\'/}"
    v_clean="${v_clean//\"/}"
    VOCALS="$(echo "${v_clean}" | xargs)"
fi

# 3. SELECT INSTRUMENTAL
echo -e "${GRN}⌨️  Select INSTRUMENTAL (Number or Path):${NC}"
read -e -r i_raw
if [[ "$i_raw" =~ ^[0-9]+$ ]] && [ "$i_raw" -le "${#FILES[@]}" ]; then
    INSTRUMENTAL="${FILES[$((i_raw-1))]}"
else
    i_clean="${i_raw//\'/}"
    i_clean="${i_clean//\"/}"
    INSTRUMENTAL="$(echo "${i_clean}" | xargs)"
fi

# 4. PROFILE SELECTION (Moved up so we know the name)
echo -e "${CYAN}--------------------------------------------------------"
echo -e "Select Mixing Profile:${NC}"
echo -e "1) ${GRN}Sing-a-long${NC}    (Vocals 1.4x | Inst 1.4x) - Standard"
echo -e "2) ${GRN}Instrumental${NC}  (Vocals 0.0x | Inst 1.4x) - No Vocals"
echo -e "3) ${GRN}Guide Track${NC}      (Vocals 0.3x | Inst 1.4x) - Quiet Vocals"
read -p ">> Choice [1-3]: " MIX_CHOICE

case $MIX_CHOICE in
    1)
        DEFAULT_NAME="song.mp3"
        FILTER="[0:a]volume=1.4[v];[1:a]volume=1.4[i];[v][i]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        ;;
    2)
        DEFAULT_NAME="instrumental.mp3"
        FILTER="[0:a]volume=0.0[v];[1:a]volume=1.4[i];[v][i]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        ;;
    3)
        DEFAULT_NAME="guide.mp3"
        FILTER="[0:a]volume=0.3[v];[1:a]volume=1.4[i];[v][i]amix=inputs=2:duration=first:normalize=0[aout]"
        ;;
    *) exit 1 ;;
esac

# 5. FILENAME PROMPT (Now pre-filled with the choice)
echo -e "${CYAN}--------------------------------------------------------${NC}"
echo -e "${GRN}Filename:${NC}"
# This will now show the default name ready for you to hit Enter
read -e -i "$DEFAULT_NAME" -r OUT_FILE

# 6. COMMAND REVIEW
BASE_CMD="ffmpeg -i \"$VOCALS\" -i \"$INSTRUMENTAL\" -filter_complex \"$FILTER\" -map \"[aout]\" -b:a 320k \"$OUT_FILE\""

echo -e "${CYAN}-----------------------------------------------------------"
echo -e "🛠️  FINAL CHECK: Edit command if needed."
echo -e "-----------------------------------------------------------${NC}"

read -e -i "$BASE_CMD" -p "🚀 RUN: " FINAL_CMD

eval "$FINAL_CMD"

if [ $? -eq 0 ]; then
    echo -e "${GRN}✅ Success! Saved to: ./$OUT_FILE${NC}"
else
    echo -e "${RED}❌ Failed.${NC}"
fi
