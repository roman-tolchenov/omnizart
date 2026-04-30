from pathlib import Path
import sys
from omnizart.cli import silence_tensorflow
from omnizart.utils import LazyLoader


music = LazyLoader("music", globals(), "omnizart.music")

this_dir = Path(__file__).parent


def transcribe_cmd(
    input_path: Path,
    model_path: Path,
    output_folder: Path,
):
    silence_tensorflow()
    music.app.transcribe(input_path, model_path, output=output_folder)


def main():
    input_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    output_folder = Path(sys.argv[3])
    output_folder.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        transcribe_cmd(input_path, model_path, output_folder)
    else:
        for input_path in input_path.glob('**/*.wav'):
            rel_path = input_path.parent.relative_to(input_path)
            out_folder = output_folder / rel_path
            transcribe_cmd(input_path, model_path, out_folder)


if __name__ == "__main__":
    main()
