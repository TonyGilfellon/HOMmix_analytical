from pathlib import Path
import numpy as np


def load_Ez_fields(field_data_fname):
    """
    Load Ez fields from a field_data.npz file.

    Returns
    -------
    dict
        {
            "E1": Ez_E1,
            "E2": Ez_E2,
            "plus": Ez_plus,
            "minus": Ez_minus,
        }

    All arrays are returned as float64 numpy arrays.
    """

    field_data_fname = Path(field_data_fname)

    with np.load(field_data_fname, allow_pickle=True) as data:

        # E1
        if "E1_Ez" in data:
            Ez_E1 = np.asarray(data["E1_Ez"], dtype=float)
        elif "Ez1" in data:
            Ez_E1 = np.asarray(data["Ez1"], dtype=float)
        else:
            raise KeyError(
                f"Could not find E1 Ez field in {field_data_fname}"
            )

        # E2
        if "E2_Ez" in data:
            Ez_E2 = np.asarray(data["E2_Ez"], dtype=float)
        elif "Ez2" in data:
            Ez_E2 = np.asarray(data["Ez2"], dtype=float)
        else:
            raise KeyError(
                f"Could not find E2 Ez field in {field_data_fname}"
            )

        # Plus
        if "Ez_plus" in data:
            Ez_plus = np.asarray(data["Ez_plus"], dtype=float)
        elif "plus_Ez" in data:
            Ez_plus = np.asarray(data["plus_Ez"], dtype=float)
        else:
            raise KeyError(
                f"Could not find plus Ez field in {field_data_fname}"
            )

        # Minus
        if "Ez_minus" in data:
            Ez_minus = np.asarray(data["Ez_minus"], dtype=float)
        elif "minus_Ez" in data:
            Ez_minus = np.asarray(data["minus_Ez"], dtype=float)
        else:
            raise KeyError(
                f"Could not find minus Ez field in {field_data_fname}"
            )

    return {
        "E1": Ez_E1,
        "E2": Ez_E2,
        "plus": Ez_plus,
        "minus": Ez_minus,
    }

if __name__ == "__main__":
    directory = r"D:\PhD\HOMmix\HOMmix_analytical\analysis\homotypic_dipoles\TM112_TM120"
    field_data_fname = f"{directory}\\field_data.npz"

    fields = load_Ez_fields(
        field_data_fname
    )

    Ez_E1 = fields["E1"]
    Ez_E2 = fields["E2"]
    Ez_plus = fields["plus"]
    Ez_minus = fields["minus"]

    print(Ez_E1.shape)
    print(Ez_plus.shape)