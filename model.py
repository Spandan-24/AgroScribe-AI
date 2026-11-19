"""
fix_model.py
------------
Fixes Keras .h5 models saved with newer TF/Keras that use the
'DTypePolicy' dict format for 'dtype' in layer configs, which
older TensorFlow versions cannot deserialize
(error: "'str' object has no attribute 'name'").

Usage:
    python fix_model.py models/agro_cnn.h5 models/agro_cnn_fixed.h5
"""

import sys
import json
import h5py


def fix_dtype_policy(config):
    """Recursively replace DTypePolicy dicts with plain dtype strings."""
    if isinstance(config, dict):
        if "dtype" in config and isinstance(config["dtype"], dict):
            dtype_cfg = config["dtype"]
            # Extract the actual dtype name, e.g. 'float32'
            inner = dtype_cfg.get("config", {})
            dtype_name = inner.get("name", "float32")
            config["dtype"] = dtype_name

        for key, value in config.items():
            if isinstance(value, (dict, list)):
                fix_dtype_policy(value)

    elif isinstance(config, list):
        for item in config:
            fix_dtype_policy(item)

    return config


def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_model.py <input.h5> <output.h5>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading model architecture from: {input_path}")

    with h5py.File(input_path, "r") as f:
        model_config_raw = f.attrs.get("model_config")
        if model_config_raw is None:
            print("ERROR: No 'model_config' attribute found in this file.")
            sys.exit(1)

        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode("utf-8")

        model_config = json.loads(model_config_raw)

    print("Patching DTypePolicy entries in layer configs...")
    fixed_config = fix_dtype_policy(model_config)
    fixed_config_str = json.dumps(fixed_config)

    print(f"Writing patched model to: {output_path}")

    # Copy the original file first (preserves weights), then patch the attr
    import shutil
    shutil.copy(input_path, output_path)

    with h5py.File(output_path, "a") as f:
        # h5py attrs have a size limit; large configs may need np.bytes_ workaround
        try:
            f.attrs.modify("model_config", fixed_config_str)
        except Exception:
            del f.attrs["model_config"]
            f.attrs.create("model_config", fixed_config_str.encode("utf-8"))

    print("Done. Try loading:", output_path)


if __name__ == "__main__":
    main()