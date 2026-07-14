from pathlib import Path
import shutil


def copy_homotypic_field_summary_pdfs(
    analysis_root: str | Path = r"D:\PhD\HOMmix\HOMmix_analytical\analysis",
    dest_dir: str | Path = r"D:\PhD\PRAB\figures",
    *,
    overwrite: bool = True,
) -> dict[str, list[Path]]:
    """
    Copy merged field-summary PDFs from homotypic mono-, di- and quadrupole analyses
    into a single PRAB figures folder.

    Expected source pattern:
        analysis_root/
            homotypic_monopoles/
                TM012_TM020/
                    slice_summary_pdfs/
                        TM012_TM020_field_summary.pdf

            homotypic_dipoles/
                TM112_TM120/
                    slice_summary_pdfs/
                        TM112_TM120_field_summary.pdf

            homotypic_quadrupoles/
                TM212_TM220/
                    slice_summary_pdfs/
                        TM212_TM220_field_summary.pdf

    Destination filenames:
        homotypic_monopole_TM012_TM020_field_summary.pdf
        homotypic_dipole_TM112_TM120_field_summary.pdf
        homotypic_quadrupole_TM212_TM220_field_summary.pdf
    """

    analysis_root = Path(analysis_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = {
        "homotypic_monopole": analysis_root / "homotypic_monopoles",
        "homotypic_dipole": analysis_root / "homotypic_dipoles",
        "homotypic_quadrupole": analysis_root / "homotypic_quadrupoles",
    }

    copied: dict[str, list[Path]] = {label: [] for label in class_dirs}
    missing_roots: list[Path] = []

    for label, class_root in class_dirs.items():
        if not class_root.exists():
            missing_roots.append(class_root)
            continue

        for crossing_dir in sorted(class_root.glob("TM*_TM*")):
            if not crossing_dir.is_dir():
                continue

            expected_pdf = (
                crossing_dir
                / "slice_summary_pdfs"
                / f"{crossing_dir.name}_field_summary.pdf"
            )

            if not expected_pdf.exists():
                print(f"Missing: {expected_pdf}")
                continue

            dest_pdf = dest_dir / f"{label}_{expected_pdf.name}"

            if dest_pdf.exists() and not overwrite:
                print(f"Exists, skipped: {dest_pdf}")
                continue

            shutil.copy2(expected_pdf, dest_pdf)
            copied[label].append(dest_pdf)
            print(f"Copied: {expected_pdf} -> {dest_pdf}")

    if missing_roots:
        print("\nMissing analysis folders:")
        for p in missing_roots:
            print(f"  {p}")

    print("\nCopy summary:")
    for label, paths in copied.items():
        print(f"  {label}: {len(paths)} PDFs")

    return copied

from pathlib import Path
import shutil
import re


def _heterotypic_dest_name(label: str, src: Path) -> str:
    """
    Convert, for example:

        label:
            heterotypic_monopole_dipole

        src.name:
            monopole_dipole__TM_032__TM_131__ell_0p77869379_field_summary.pdf

    into:

        heterotypic_monopole_dipole_TM_032__TM_131__field_summary.pdf
    """
    stem = src.stem

    pattern = (
        r"^(monopole_dipole|monopole_quadrupole|dipole_quadrupole)"
        r"__(TM_\d+)"
        r"__(TM_\d+)"
        r"__ell_[^_]+_field_summary$"
    )

    match = re.match(pattern, stem)

    if match:
        pair_type, mode_i, mode_j = match.groups()
        return f"heterotypic_{pair_type}_{mode_i}__{mode_j}__field_summary.pdf"

    # Fallback: strip any __ell_... section if present.
    stem_no_ell = re.sub(r"__ell_.*?_field_summary$", "__field_summary", stem)
    if stem_no_ell != stem:
        return f"{label}_{stem_no_ell}.pdf"

    return f"{label}_{src.name}"


def copy_all_summary_pdfs_to_prab_figures(
    analysis_root: str | Path = r"D:\PhD\HOMmix\HOMmix_analytical\analysis",
    dest_dir: str | Path = r"D:\PhD\PRAB\figures",
    *,
    overwrite: bool = True,
) -> dict[str, list[Path]]:

    analysis_root = Path(analysis_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, list[Path]] = {
        "homotypic_monopole": [],
        "homotypic_dipole": [],
        "homotypic_quadrupole": [],
        "heterotypic_monopole_dipole": [],
        "heterotypic_monopole_quadrupole": [],
        "heterotypic_dipole_quadrupole": [],
    }

    # ------------------------------------------------------------------
    # Homotypic summaries
    # ------------------------------------------------------------------
    homotypic_dirs = {
        "homotypic_monopole": analysis_root / "homotypic_monopoles",
        "homotypic_dipole": analysis_root / "homotypic_dipoles",
        "homotypic_quadrupole": analysis_root / "homotypic_quadrupoles",
    }

    for label, root in homotypic_dirs.items():
        if not root.exists():
            print(f"Missing: {root}")
            continue

        for crossing_dir in sorted(root.glob("TM*_TM*")):
            src = (
                crossing_dir
                / "slice_summary_pdfs"
                / f"{crossing_dir.name}_field_summary.pdf"
            )

            if not src.exists():
                print(f"Missing: {src}")
                continue

            dst = dest_dir / f"{label}_{src.name}"

            if dst.exists() and not overwrite:
                print(f"Exists, skipped: {dst}")
                continue

            shutil.copy2(src, dst)
            copied[label].append(dst)
            print(f"Copied: {src} -> {dst}")

    # ------------------------------------------------------------------
    # Heterotypic summaries
    # ------------------------------------------------------------------
    heterotypic_root = analysis_root / "heterotypic_crossings"

    heterotypic_dirs = {
        "heterotypic_monopole_dipole": heterotypic_root / "monopole_dipole",
        "heterotypic_monopole_quadrupole": heterotypic_root / "monopole_quadrupole",
        "heterotypic_dipole_quadrupole": heterotypic_root / "dipole_quadrupole",
    }

    for label, root in heterotypic_dirs.items():
        if not root.exists():
            print(f"Missing: {root}")
            continue

        for crossing_dir in sorted(root.iterdir()):
            if not crossing_dir.is_dir():
                continue

            src_dir = crossing_dir / "slice_summary_pdfs"

            if not src_dir.exists():
                print(f"Missing: {src_dir}")
                continue

            pdfs = sorted(src_dir.glob("*_field_summary.pdf"))

            if not pdfs:
                print(f"No summary PDF found in: {src_dir}")
                continue

            for src in pdfs:
                dst = dest_dir / _heterotypic_dest_name(label, src)

                if dst.exists() and not overwrite:
                    print(f"Exists, skipped: {dst}")
                    continue

                shutil.copy2(src, dst)
                copied[label].append(dst)
                print(f"Copied: {src} -> {dst}")

    print("\nCopy summary:")
    for label, paths in copied.items():
        print(f"  {label}: {len(paths)} PDFs")

    return copied

if __name__ == "__main__":


    # copied = copy_homotypic_field_summary_pdfs()
    copied = copy_all_summary_pdfs_to_prab_figures()