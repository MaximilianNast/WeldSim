import numpy as np
import scipy.optimize as spopt


def calculate_height_and_width(v_wire_feed,
                               v_weld,
                               d_wire,
                               u_weld,
                               i_weld,
                               t_0,

                               k,
                               t_m,
                               alpha,
                               gamma,
                               rho,
                               g,

                               eta_weld = 0.8
                               ):
    """
    Calculate height and width of a weld bead based on given parameters, using "Analytical process model for wire + arc
    additive manufacturing" (Sergio Rios et al. - 2018)
    https://doi.org/10.1016/j.addma.2018.04.003
    https://www.sciencedirect.com/science/article/pii/S2214860417303470)

    process parameters
    :param float v_wire_feed: wire feed speed [m/min]
    :param float v_weld: welding speed [cm/min]
    :param float d_wire: the diameter of the wire [mm]
    :param float u_weld: weld arc amperage [A]
    :param float i_weld: weld arc voltage [V]
    :param float t_0: interpass temperature [°C]

    material parameters
    :param float k: thermal conductivity [W/(m K)]
    :param flaot t_m: melting point [°C]
    :param float alpha: thermal diffusivity [m²/s]
    :param float gamma: surface tension [N/m]
    :param float rho: density [kg/m³]
    :param float g: gravitational acceleration [m/s²]

    optional parameters
    :param eta_weld: arc efficiency factor [-]

    :return: layer height
    :rtype: int
    :raises Nothing. just be careful ig ;)
    """

    v_wire_feed = v_wire_feed*1000/60  # [mm/s]
    v_weld = v_weld*10/60  # [mm/s]

    # 1. Querschnittsflaeche
    Awire = 0.25 * np.pi * d_wire**2       # [mm²]
    CSA   = v_wire_feed * Awire/v_weld                # [mm²]
    # print('CSA   = ' + str(round(CSA,2)) + ' mm²')

    # 2. Waermeeintrag
    Qwaam = eta_weld * u_weld * i_weld       # [W]

    # 3. Geometrie (theta & r)
    # Berechnung mit Gleichung (23) aus der Literaturstelle
    V = Qwaam/(8*k*(t_m - t_0))
    W = v_weld/(2*alpha*1000)
    CSA = CSA/(1000*1000)                # [mm²] <-- !

    def f(x):
        return V - np.sqrt(CSA/(x+np.sin(x))) * np.cos(x/2) * (
            1/5 + W * np.sqrt(CSA/(x+np.sin(x))) * (np.sin(x/2) + np.pi/4))

    res     = spopt.root(f, [1])
    theta   = res.x[0]
    r       = np.sqrt(CSA*1000*1000/(theta+np.sin(theta)))  # [mm]
    # print('theta = ' + str(round(theta,3)) + '° & r = ' + str(round(r,2)) + ' mm')

    # 4. LH, EWW & R
    thetaDeg = theta * 180 / np.pi  # Winkel [DEG]
    LH = 2 * r * np.sin(theta / 2)  # [mm]
    EWW = 2 * r * np.cos(theta / 2)  # [mm]
    R = r * (1 - np.sin(theta / 2))  # Radius [mm]
    # print('LH    = ' + str(round(LH, 2)) + ' mm')
    # print('EWW   = ' + str(round(EWW, 2)) + ' mm')

    # 5. Relationsgleichung pruefen
    kappa = np.sqrt(gamma / (rho * g)) * 1000  # KappilarRadius [mm]
    if r * np.sqrt(1 + np.sin(theta / 2)) <= kappa:
        pass
        # print('Relationsgleichung erfuellt!\nBerechnung wird durchgefuehrt.')
    else:
        pass
        # print('Relationsgleichung nicht erfuellt!\nBitte neue Startparameter EWW & LH waehlen!')

    # 6. Wanddicke
    WW = 2 * r  # [mm]
    # print('WW    = ' + str(round(WW, 2)) + ' mm')

    # 7. Einschweißtiefe
    aWD = (WW + LH) / 2 - r * (1 - 0.25 * np.pi)  # [mm]
    # print('aWD   = ' + str(round(aWD, 2)) + ' mm')

    return LH, EWW


if __name__ == "__main__":
    print(calculate_height_and_width(6.7, 600.0/10, 1.0, 165.0, 12.5, 130, 32, 1577, 32/(7200*700), 1.2, 7200, 9.81, 0.8))
