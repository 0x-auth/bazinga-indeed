#!/bin/bash
REMOTE="mygdrive:"

echo "------------------------------------------------"
echo "🕵️ SCANNING FOR REMAINING FOLDER-CEPTION..."
echo "------------------------------------------------"

# 1. Pulling files out of folders that match their own name
rclone lsf "$REMOTE" -R --files-only | while read -r FULLPATH; do
    FILENAME=$(basename "$FULLPATH")
    DIRNAME=$(dirname "$FULLPATH")
    PARENT=$(basename "$DIRNAME")
    GRANDPARENT=$(dirname "$DIRNAME")

    # This targets files where the parent folder name is similar to the filename
    # Example: "π (2).docx/π (2).docx"
    if [[ "$FILENAME" == "$PARENT"* ]] || [[ "$PARENT" == *"$FILENAME"* ]]; then
        echo "🎯 Target Found: $FULLPATH"
        
        if [[ "$GRANDPARENT" == "." ]]; then
            TARGET="$FILENAME"
        else
            TARGET="$GRANDPARENT/$FILENAME"
        fi
        
        # We use --ignore-existing to not overwrite the Google Docs you just made
        rclone move "${REMOTE}${FULLPATH}" "${REMOTE}${TARGET}" --ignore-existing -v
    fi
done

echo "------------------------------------------------"
echo "💥 CLEANING WEIRD CHARACTERS FROM ALL FILENAMES"
echo "------------------------------------------------"
# This specifically targets the #, π, →, and spaces left in the WHOLE drive
rclone lsf "$REMOTE" -R --files-only | grep -E '[[:space:]]|#|→|π|⊕' | while read -r WEIRD_PATH; do
    DIR=$(dirname "$WEIRD_PATH")
    BASE=$(basename "$WEIRD_PATH")
    
    # Clean the name: Remove spaces and special chars
    CLEAN_BASE=$(echo "$BASE" | sed 's/[[:space:]]/_/g' | sed 's/[#→π⊕]//g' | sed 's/[(|)]//g')
    
    if [[ "$DIR" == "." ]]; then
        NEW_PATH="$CLEAN_BASE"
    else
        NEW_PATH="$DIR/$CLEAN_BASE"
    fi
    
    echo "Fixing Name: $BASE -> $CLEAN_BASE"
    rclone move "${REMOTE}${WEIRD_PATH}" "${REMOTE}${NEW_PATH}" --ignore-existing --quiet
done

echo "------------------------------------------------"
echo "🗑️ FINAL PURGE OF EMPTY FOLDERS"
echo "------------------------------------------------"
rclone rmdirs "$REMOTE" --leave-root -v

echo "✅ System 100% Linear and Clean, Bhai!"
