import math
from numba import njit


@njit(cache=True)
def _get_piecewise_position(p0_x, p0_y, v_mag, alpha, w_roll, w_dir, t, mu_s, mu_r, g, R):
    vx = v_mag * math.cos(alpha)
    vy = v_mag * math.sin(alpha)

    # Spin relative to global axes, locked to the aiming angle (alpha)
    w_x = -w_roll * math.sin(alpha) + w_dir * math.cos(alpha)
    w_y = w_roll * math.cos(alpha) + w_dir * math.sin(alpha)

    ux = vx - R * w_y
    uy = vy + R * w_x
    u_mag = math.hypot(ux, uy)

    if u_mag < 1e-6:
        t_s = 0.0
        u_hat_x, u_hat_y = 0.0, 0.0
    else:
        u_hat_x = ux / u_mag
        u_hat_y = uy / u_mag
        t_s = (2.0 / 7.0) * u_mag / (mu_s * g)

    if t <= t_s:

        accel_mag = 0.5 * mu_s * g
        px = p0_x + vx * t - accel_mag * u_hat_x * (t ** 2)
        py = p0_y + vy * t - accel_mag * u_hat_y * (t ** 2)
        return px, py
    else:

        accel_mag = 0.5 * mu_s * g
        p_s_x = p0_x + vx * t_s - accel_mag * u_hat_x * (t_s ** 2)
        p_s_y = p0_y + vy * t_s - accel_mag * u_hat_y * (t_s ** 2)

        v_s_x = vx - mu_s * g * u_hat_x * t_s
        v_s_y = vy - mu_s * g * u_hat_y * t_s
        v_s_mag = math.hypot(v_s_x, v_s_y)

        if v_s_mag < 1e-6:
            return p_s_x, p_s_y

        v_s_hat_x = v_s_x / v_s_mag
        v_s_hat_y = v_s_y / v_s_mag

        t_roll = t - t_s

        # Prevent the ball from accelerating backward once it stops!
        t_stop = v_s_mag / (mu_r * g)
        if t_roll > t_stop:
            t_roll = t_stop

        r_accel_mag = 0.5 * mu_r * g

        px = p_s_x + v_s_x * t_roll - r_accel_mag * v_s_hat_x * (t_roll ** 2)
        py = p_s_y + v_s_y * t_roll - r_accel_mag * v_s_hat_y * (t_roll ** 2)

        return px, py


@njit(cache=True)
def solve_exact_aim_angle(p0_x, p0_y, p1_x, p1_y, v_mag, w_roll, w_dir, mu_s, mu_r, g, R, max_iter=30, tol=1e-4):
    dx = p1_x - p0_x
    dy = p1_y - p0_y
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        return 0.0, True

    base_alpha = math.atan2(dy, dx)
    alpha = base_alpha

    if v_mag < 1e-6:
        return alpha, False

    t = dist / v_mag
    delta = 1e-6

    for _ in range(max_iter):
        px, py = _get_piecewise_position(p0_x, p0_y, v_mag, alpha, w_roll, w_dir, t, mu_s, mu_r, g, R)

        Fx = px - p1_x
        Fy = py - p1_y

        if math.hypot(Fx, Fy) < tol:
            return alpha, True

        px_a, py_a = _get_piecewise_position(p0_x, p0_y, v_mag, alpha + delta, w_roll, w_dir, t, mu_s, mu_r, g, R)
        dFx_da = (px_a - px) / delta
        dFy_da = (py_a - py) / delta

        px_t, py_t = _get_piecewise_position(p0_x, p0_y, v_mag, alpha, w_roll, w_dir, t + delta, mu_s, mu_r, g, R)
        dFx_dt = (px_t - px) / delta
        dFy_dt = (py_t - py) / delta

        det = (dFx_da * dFy_dt) - (dFx_dt * dFy_da)
        if abs(det) < 1e-12:
            break

        step_alpha = (dFy_dt * (-Fx) - dFx_dt * (-Fy)) / det
        step_t = (dFx_da * (-Fy) - dFy_da * (-Fx)) / det

        # The Clamp
        if step_alpha > 0.05:
            step_alpha = 0.05
        elif step_alpha < -0.05:
            step_alpha = -0.05

        alpha += step_alpha
        t += step_t

        if t <= 0:
            t = 0.01

        if abs(alpha - base_alpha) > 0.52:
            return base_alpha, False

    return alpha, False


@njit(cache=True)
def check_path_obstruction(start_x, start_y, end_x, end_y, circle_x, circle_y, circle_radius, path_radius):
    """
    Checks if a circle obstructs a sweeping path between two points.
    Returns True if obstructed, False if clear.
    """
    dx = end_x - start_x
    dy = end_y - start_y

    line_len_sq = dx * dx + dy * dy

    # If the path is essentially a single point
    if line_len_sq < 1e-8:
        dist_sq = (circle_x - start_x) ** 2 + (circle_y - start_y) ** 2
        return dist_sq < (circle_radius + path_radius) ** 2

    # Project the circle's center onto the line segment to find the closest point
    t = ((circle_x - start_x) * dx + (circle_y - start_y) * dy) / line_len_sq

    # Clamp t to [0.0, 1.0] so we only check the line segment, not the infinite line
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    closest_x = start_x + t * dx
    closest_y = start_y + t * dy

    # Check distance from the circle's center to the closest point on the path
    dist_sq = (circle_x - closest_x) ** 2 + (circle_y - closest_y) ** 2

    # If the distance is less than the sum of the radii, they collide
    return dist_sq < (circle_radius + path_radius) ** 2


@njit(cache=True)
def check_escape_rays_numba(gb_x, gb_y, obstacles, num_rays, escape_dist, path_radius):
    """
    Blasts outward rays from a coordinate.
    obstacles: Nx3 array of (x, y, radius)
    Returns True if at least one ray escapes without hitting an obstacle.
    """
    angle_step = (2.0 * math.pi) / num_rays

    for i in range(num_rays):
        angle = i * angle_step
        end_x = gb_x + math.cos(angle) * escape_dist
        end_y = gb_y + math.sin(angle) * escape_dist

        dx = end_x - gb_x
        dy = end_y - gb_y
        line_len_sq = dx * dx + dy * dy

        if line_len_sq < 1e-8:
            continue

        ray_blocked = False

        for j in range(obstacles.shape[0]):
            ox = obstacles[j, 0]
            oy = obstacles[j, 1]
            orad = obstacles[j, 2]

            t = ((ox - gb_x) * dx + (oy - gb_y) * dy) / line_len_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            closest_x = gb_x + t * dx
            closest_y = gb_y + t * dy

            dist_sq = (ox - closest_x) ** 2 + (oy - closest_y) ** 2

            if dist_sq < (orad + path_radius) ** 2:
                ray_blocked = True
                break  # Ray failed, move immediately to the next angle

        if not ray_blocked:
            return True  # A completely clear path exists!

    return False  # Walled in completely