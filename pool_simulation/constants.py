INCHES_TO_M = 0.0254
SQRT_3_INCHES = 3**0.5*INCHES_TO_M
BLACK_SPOT_X = 0.481596438273  # m
PIXELS_PER_METER = 500
CUE_BALL_RADIUS = 0.0238125  # m
OBJECT_BALL_RADIUS = 0.0254  # m
CUE_BALL_MASS = 0.097  # kg
OBJECT_BALL_MASS = 0.118  # kg
TABLE_WIDTH = 1.8288  # m
TABLE_HEIGHT = 0.9144  # m
CUSHION_WIDTH = 0.06  # m
CUSHION_HEIGHT_EFF = 0.032
CUSHION_HEIGHT_ACTUAL = 0.038
MIDDLE_POCKET_X = 0.0
MIDDLE_POCKET_Y = 0.507
CORNER_POCKET_X = 0.9441
CORNER_POCKET_Y = 0.4869
POCKET_RADIUS = 0.051  # m
CUE_RADIUS = 0.005  # m
CUE_LENGTH = 1.45  # m
ALPHA_C = 0.644685359294  # rad
ALPHA_M = 0.867338336423  # rad
MU_S = 0.24
MU_R = 0.03
MU_SP = 0.02
MU_B = 0.04
E_C = 0.82  # Cushion coefficient of restitution
MU_C = 0.2  # Cushion friction coefficient (grippiness of the cloth on the rubber)
K_N = 1e4  # Cushion normal spring stiffness
BETA_N = 1.0  # Normal mass-matrix coefficient
BETA_T = 3.5  # Tangential mass-matrix coefficient (1 + mR^2/I = 1 + 2.5 = 3.5)
g = 9.80665
RESTITUTION = 1.0
MAX_SPEED = 10  # m/s
FPS = 60
COLOUR_MAP = {
    0: (255, 255, 255),  # Cue ball
    1: (200, 30, 30),  # Red Ball
    2: (240, 240, 40),  # Yellow Ball
    3: (20, 20, 20),  # Black Ball

# --- Debug Colours ---
    -1: (0, 255, 255),   # Cyan
    -2: (255, 0, 255),   # Magenta
    -3: (255, 128, 0),   # Orange
    -4: (128, 0, 255),   # Neon Purple
    -5: (0, 100, 255),   # Bright Blue
    -6: (143, 255, 188)  # Light Green-Blue
}
COLOUR_NAMES = {
    1: "Red",
    2: "Yellow"
}