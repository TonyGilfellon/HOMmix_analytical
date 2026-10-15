from pathlib import Path

ROOT = Path(r"D:\PhD\NIMA\submission\pre_submission_files")
TEX_FILE = ROOT / "main.tex"

REPLACEMENTS = {
    "heterotypic_monopole_dipole_": "het_MD_",
    "heterotypic_monopole_quadrupole_": "het_MQ_",
    "heterotypic_dipole_quadrupole_": "het_DQ_",
    "homotypic_monopole_": "hom_MM_",
    "homotypic_dipole_": "hom_DD_",
    "homotypic_quadrupole_": "hom_QQ_",
}

# ------------------------------------------------------------------
# Read main.tex
# ------------------------------------------------------------------

tex = TEX_FILE.read_text(encoding="utf-8")

renamed = []

# ------------------------------------------------------------------
# Rename field-summary PDFs and update main.tex
# ------------------------------------------------------------------

for old_path in ROOT.glob("*_field_summary.pdf"):

    old_name = old_path.name
    new_name = old_name

    for old_prefix, new_prefix in REPLACEMENTS.items():
        if new_name.startswith(old_prefix):
            new_name = new_name.replace(old_prefix, new_prefix, 1)
            break

    # Remove the redundant "__" between mode names.
    new_name = new_name.replace("__", "_")

    # Remove the redundant field-summary suffix.
    new_name = new_name.replace("_field_summary.pdf", ".pdf")

    if new_name == old_name:
        print(f"SKIPPED: {old_name}")
        continue

    new_path = ROOT / new_name

    if new_path.exists():
        raise FileExistsError(
            f"Cannot rename:\n"
            f"  {old_name}\n"
            f"to:\n"
            f"  {new_name}\n"
            f"because the destination already exists."
        )

    # Update every reference in main.tex.
    tex = tex.replace(old_name, new_name)

    # Rename the actual file.
    old_path.rename(new_path)

    renamed.append((old_name, new_name))

# ------------------------------------------------------------------
# Write modified main.tex
# ------------------------------------------------------------------

TEX_FILE.write_text(tex, encoding="utf-8")

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

print()
print(f"Renamed {len(renamed)} files:")
print()

for old_name, new_name in renamed:
    print(f"{old_name}")
    print(f"    -> {new_name}")

print()
print("main.tex references updated.")