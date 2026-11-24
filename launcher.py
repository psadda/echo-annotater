"""Entry point for packaged builds (e.g. PyInstaller), which need a plain script rather than a console command."""

from echo_annotator import main

if __name__ == "__main__":
    main()
