#!/usr/bin/env python
"""
fix_encoding.py - Run this BEFORE main.py to fix Windows Unicode errors
"""
import sys
import io
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    print("[Fix] Applying UTF-8 encoding fix for Windows...")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    print("[Fix] Encoding fixed successfully!")

# Now import and run main
if __name__ == '__main__':
    import main
    main.main()