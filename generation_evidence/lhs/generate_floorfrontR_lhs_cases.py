from pathlib import Path

import generate_floorfrontdriver_lhs_cases as lhs
import generate_floorfrontdriver_random_cases as base


lhs.TEMPLATE = Path("floor_panel_largest_2_pid_2000395_133_floorfrontR_with_ball.key")
lhs.OUTPUT_ROOT = Path("cases_floorfrontR_lhs_500")
lhs.DESIGN_NAME = "floorfrontR"
lhs.DEFAULT_CASES = 500
lhs.DEFAULT_SEED = 20260728
lhs.PANEL_PID = 2000395

# The shared template parser selects valid panel elements through this module global.
base.PANEL_PID = lhs.PANEL_PID


if __name__ == "__main__":
    lhs.main()
