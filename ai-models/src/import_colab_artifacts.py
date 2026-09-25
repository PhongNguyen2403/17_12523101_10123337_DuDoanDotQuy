"""Import model artifacts exported from 03_train.ipynb."""

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


REQUIRED_FILES = {
    "model.joblib",
    "schema.json",
    "metadata.json",
}


def import_artifacts(archive_path: Path, models_dir: Path) -> None:
    with ZipFile(archive_path) as archive:
        files = {}
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or len(path.parts) < 2 or path.parts[0] != "models":
                continue
            relative = Path(*path.parts[1:])
            files[relative.as_posix()] = member

        missing = REQUIRED_FILES - {Path(name).name for name in files}
        if missing:
            raise FileNotFoundError(
                "Artifact thiếu trong file ZIP: " + ", ".join(sorted(missing))
            )

        models_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = models_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        for relative_name, member in files.items():
            relative_path = Path(relative_name)
            destination = candidates_dir / relative_path.name if relative_path.parts[0] == "candidates" else models_dir / relative_path.name
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())

    print(f"Imported model artifacts into: {models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to stroke_models.zip")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Destination model directory",
    )
    args = parser.parse_args()
    import_artifacts(args.archive, args.models_dir)
