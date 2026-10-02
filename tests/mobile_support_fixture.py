"""Explicit software-only solved support for socket/control lifecycle tests."""


def support_contract():
    return dict(version=1, model_identity_sha256='e'*64,
                robot_shapes=['/Fixture/foot', '/Fixture/trunk'], foot_shapes=['/Fixture/foot'],
                ground_shapes=['/Fixture/ground'], gravity_world_m_s2=[0., 0., -9.81])


def support(step, sim_time_s):
    # Synthetic evidence, not a physics producer or acceptance measurement.
    return dict(version=1, status='known', reason='', step=step, sim_time_s=sim_time_s,
        contacts=[dict(shape_a_id=0, shape_b_id=1, shape_a='/Fixture/ground', shape_b='/Fixture/foot',
            force_on_b_world_n=[0., 0., 3.], normal_force_n=3., normal_a_to_b_world=[0., 0., 1.],
            point_world_m=[0., 0., 0.])])
