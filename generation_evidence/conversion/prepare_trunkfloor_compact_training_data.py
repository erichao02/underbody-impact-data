from pathlib import Path

import prepare_floorfrontR_compact_training_data as compact


compact.DEFAULT_CASE_ROOT = Path("cases_trunkfloor_random_50")
compact.DEFAULT_OUT_ROOT = Path("training_data_trunkfloor_compact")
compact.DEFAULT_PANEL_KEY = Path("floor_panel_largest_1_pid_2000447_153_trunkfloor.key")
compact.PANEL_PID = 2000447


if __name__ == "__main__":
    compact.main()
