from pathlib import Path

import prepare_floorfrontR_compact_training_data as compact


compact.DEFAULT_CASE_ROOT = Path("cases_floorfrontdriver_random_50")
compact.DEFAULT_OUT_ROOT = Path("training_data_floorfrontdriver_compact")
compact.DEFAULT_PANEL_KEY = Path("floor_panel_largest_3_pid_2000394_373_floorfrontdriver.key")
compact.PANEL_PID = 2000394


if __name__ == "__main__":
    compact.main()
