from __future__ import annotations

from pathlib import Path
import re
import shutil


def _copy_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"Exists, skipped: {destination}")
        return False

    shutil.copy2(source, destination)
    print(f"Copied: {source} -> {destination}")
    return True


def _homotypic_destination_name(
    class_label: str,
    crossing_dir: Path,
) -> str:
    """Return the PRAB filename for a unified homotypic crossing.

    Example
    -------
    class_label = "homotypic_monopole"
    crossing_dir.name = "TM012_TM021"

    returns
        homotypic_monopole_TM012_TM021_field_summary.pdf
    """
    return (
        f"{class_label}_{crossing_dir.name}_field_summary.pdf"
    )


def _heterotypic_destination_name(
    class_label: str,
    source: Path,
) -> str:
    """Remove the crossing-length string from heterotypic summary names."""
    stem = source.stem

    pattern = (
        r"^(monopole_dipole|monopole_quadrupole|dipole_quadrupole)"
        r"__(TM_\d+)"
        r"__(TM_\d+)"
        r"__ell_[^_]+_field_summary$"
    )
    match = re.match(pattern, stem)

    if match:
        pair_type, mode_i, mode_j = match.groups()
        return (
            f"heterotypic_{pair_type}_"
            f"{mode_i}__{mode_j}__field_summary.pdf"
        )

    stem_without_ell = re.sub(
        r"__ell_.*?_field_summary$",
        "__field_summary",
        stem,
    )
    if stem_without_ell != stem:
        return f"{class_label}_{stem_without_ell}.pdf"

    return f"{class_label}_{source.name}"


def copy_all_summary_pdfs_to_prab_figs(
    analysis_root: str | Path = (
        r"D:\PhD\HOMmix\HOMmix_analytical\analysis"
    ),
    destination_dir: str | Path = r"D:\PhD\PRAB\figs",
    *,
    overwrite: bool = True,
) -> dict[str, list[Path]]:
    """Copy and rename homotypic and heterotypic appendix summary PDFs.

    Unified homotypic source layout
    --------------------------------
    analysis/
        homotypic_rf_multipole/
            monopole_monopole/
                TM012_TM021/
                    slice_summary_pdfs/
                        TM012_TM021_field_summary.pdf
            dipole_dipole/
            quadrupole_quadrupole/

    Destination examples
    --------------------
        homotypic_monopole_TM012_TM021_field_summary.pdf
        homotypic_dipole_TM112_TM120_field_summary.pdf
        homotypic_quadrupole_TM213_TM222_field_summary.pdf

    Heterotypic source layout remains
    ---------------------------------
    analysis/
        heterotypic_crossings/
            monopole_dipole/
            monopole_quadrupole/
            dipole_quadrupole/
    """
    analysis_root = Path(analysis_root)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, list[Path]] = {
        "homotypic_monopole": [],
        "homotypic_dipole": [],
        "homotypic_quadrupole": [],
        "heterotypic_monopole_dipole": [],
        "heterotypic_monopole_quadrupole": [],
        "heterotypic_dipole_quadrupole": [],
    }

    # ------------------------------------------------------------------
    # Unified homotypic summaries
    # ------------------------------------------------------------------
    homotypic_root = analysis_root / "homotypic_rf_multipole"
    homotypic_directories = {
        "homotypic_monopole": (
            homotypic_root / "monopole_monopole"
        ),
        "homotypic_dipole": (
            homotypic_root / "dipole_dipole"
        ),
        "homotypic_quadrupole": (
            homotypic_root / "quadrupole_quadrupole"
        ),
    }

    for class_label, class_root in homotypic_directories.items():
        if not class_root.exists():
            print(f"Missing homotypic root: {class_root}")
            continue

        for crossing_dir in sorted(class_root.glob("TM*_TM*")):
            if not crossing_dir.is_dir():
                continue

            source = (
                crossing_dir
                / "slice_summary_pdfs"
                / f"{crossing_dir.name}_field_summary.pdf"
            )

            if not source.exists():
                print(f"Missing homotypic summary: {source}")
                continue

            destination = (
                destination_dir
                / _homotypic_destination_name(
                    class_label,
                    crossing_dir,
                )
            )

            if _copy_file(
                source,
                destination,
                overwrite=overwrite,
            ):
                copied[class_label].append(destination)

    # ------------------------------------------------------------------
    # Heterotypic summaries
    # ------------------------------------------------------------------
    heterotypic_root = (
        analysis_root / "heterotypic_crossings"
    )
    heterotypic_directories = {
        "heterotypic_monopole_dipole": (
            heterotypic_root / "monopole_dipole"
        ),
        "heterotypic_monopole_quadrupole": (
            heterotypic_root / "monopole_quadrupole"
        ),
        "heterotypic_dipole_quadrupole": (
            heterotypic_root / "dipole_quadrupole"
        ),
    }

    for class_label, class_root in heterotypic_directories.items():
        if not class_root.exists():
            print(f"Missing heterotypic root: {class_root}")
            continue

        for crossing_dir in sorted(class_root.iterdir()):
            if not crossing_dir.is_dir():
                continue

            source_dir = crossing_dir / "slice_summary_pdfs"
            if not source_dir.exists():
                print(f"Missing heterotypic PDF folder: {source_dir}")
                continue

            pdfs = sorted(
                source_dir.glob("*_field_summary.pdf")
            )
            if not pdfs:
                print(
                    f"No heterotypic summary PDF found in: "
                    f"{source_dir}"
                )
                continue

            for source in pdfs:
                destination = (
                    destination_dir
                    / _heterotypic_destination_name(
                        class_label,
                        source,
                    )
                )
                if _copy_file(
                    source,
                    destination,
                    overwrite=overwrite,
                ):
                    copied[class_label].append(destination)

    print("\nCopy summary:")
    for class_label, paths in copied.items():
        print(f"  {class_label}: {len(paths)} PDFs")

    return copied


if __name__ == "__main__":
    copy_all_summary_pdfs_to_prab_figs(
        analysis_root=(
            r"D:\PhD\HOMmix\HOMmix_analytical\analysis"
        ),
        destination_dir=r"D:\PhD\PRAB\figs",
        overwrite=True,
    )
