import numpy as np
from . import initialization
from .bbox import Box


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
        origin = component.bbox.origin
    else:
        model = component
        origin = 0

    return tuple(np.array(np.unravel_index(np.argmax(model), model.shape)) + origin)


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
    else:
        model = component

    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
        origin = component.bbox.origin
    else:
        model = component
        origin = 0

    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + origin


def snr(component, observations):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model

    observations: `scarlet.Observation` or list thereof
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if hasattr(component, "get_model"):
        frame = None
        if not prerender:
            frame = observations[0].model_frame
        model = component.get_model(frame=frame)
        bbox = component.bbox
    else:
        model = component
        bbox = Box(model.shape)

    M = []
    W = []
    var = []
    # convolve model for every observation;
    # flatten in channel direction because it may not have all C channels; concatenate
    # do same thing for noise variance
    for obs in observations:
        model_ = obs.render(model)
        M.append(model_.reshape(-1))
        W.append((model_ / (model_.sum(axis=(-2, -1))[:, None, None])).reshape(-1))
        noise_var = obs.noise_rms ** 2
        var.append(noise_var.reshape(-1))
    M = np.concatenate(M)
    W = np.concatenate(W)
    var = np.concatenate(var)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (M * W).sum() / np.sqrt(((var * W) * W).sum())

    return snr


# adapted from https://github.com/pmelchior/shapelens/blob/master/src/Moments.cc
def moments(component, N=2, centroid=None, weight=None):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    N: int >=0
        Moment order
    centroid: array
        2D coordinate in frame of `component`
    weight: array
        weight function with same shape as `component`
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
    else:
        model = component

    if weight is None:
        weight = 1
    else:
        assert model.shape == weight.shape

    if centroid is None:
        centroid = np.array(model.shape) // 2

    grid_x, grid_y = np.indices(model.shape[-2:], dtype=np.float64)
    if len(model.shape) == 3:
        grid_y = grid_y[None, :, :]
        grid_x = grid_x[None, :, :]
    grid_y -= centroid[0]
    grid_x -= centroid[1]

    M = dict()
    for n in range(N + 1):
        for m in range(n + 1):
            # moments ordered by power in y, then x
            M[m, n - m] = (grid_y ** m * grid_x ** (n - m) * model * weight).sum(
                axis=(-2, -1)
            )
    return M


def raw_moment(data, i_order, j_order, weight):
    n_depth, n_row, n_col = data.shape
    y, x = np.mgrid[:n_row, :n_col]
    if weight is None:
        data = data * x**i_order * y**j_order
    else:
        data = data * weight * x**i_order * y**j_order
    return np.sum(data, axis=(1, 2))


def g1g2(model):
    weight = None
    if len(model.shape) == 2:
        model = model[None, :, :]

    # zeroth-order moment: total flux
    w00 = raw_moment(model, 0, 0, weight)

    # first-order moment: centroid
    w10 = raw_moment(model, 1, 0, weight)
    w01 = raw_moment(model, 0, 1, weight)
    x_c = w10 / w00
    y_c = w01 / w00

    # second-order moment: b/a ratio and position angle
    m11 = raw_moment(model, 1, 1, weight) / w00 - x_c * y_c
    m20 = raw_moment(model, 2, 0, weight) / w00 - x_c**2
    m02 = raw_moment(model, 0, 2, weight) / w00 - y_c**2

    g1 = (m20 - m02) / (m20 + m02 + 2 * np.sqrt(m20 * m02 - m11**2))
    g2 = (2 * m11) / (m20 + m02 + 2 * np.sqrt(m20 * m02 - m11**2))

    return (g1[0], g2[0])


def q_pa(model):
    weight = None
    if len(model.shape) == 2:
        model = model[None, :, :]

    # zeroth-order moment: total flux
    w00 = raw_moment(model, 0, 0, weight)

    # first-order moment: centroid
    w10 = raw_moment(model, 1, 0, weight)
    w01 = raw_moment(model, 0, 1, weight)
    x_c = w10 / w00
    y_c = w01 / w00

    # second-order moment: b/a ratio and position angle
    m11 = raw_moment(model, 1, 1, weight) / w00 - x_c * y_c
    m20 = raw_moment(model, 2, 0, weight) / w00 - x_c**2
    m02 = raw_moment(model, 0, 2, weight) / w00 - y_c**2
    cov = np.array([m20, m11, m11, m02]).T.reshape(-1, 2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # q = b/a
    q = np.min(eigvals, axis=1) / np.max(eigvals, axis=1)  # don't take sqrt

    # position angle PA: between the major axis and the east (positive x-axis)
    major_axis = eigvecs[np.arange(
        len(eigvecs)), np.argmax(eigvals, axis=1), :]
    sign = np.sign(major_axis[:, 1])  # sign of y-component
    pa = np.rad2deg(np.arccos(np.dot(major_axis, [1, 0])))
    pa = np.array([x - 180 if abs(x) > 90 else x for x in pa])
    pa *= sign
    pa = np.deg2rad(pa)

    return (q, pa)


def flux_radius(model, frac=0.5):
    """
    Determine the radius R (in pixels, along semi-major axis), 
    the flux within R has a fraction of `frac` over the total flux.

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation: 

    frac: float
        fraction of lights within this R.

    """
    from scipy.interpolate import interp1d, UnivariateSpline
    import sep

    if len(model.shape) == 2:
        model = model[None, :, :]

    w00 = raw_moment(model, 0, 0, None)
    w10 = raw_moment(model, 1, 0, None)
    w01 = raw_moment(model, 0, 1, None)
    x_cen = (w10 / w00)[0]
    y_cen = (w01 / w00)[0]

    q, pa = q_pa(model)

    total_flux = model.sum(axis=(1, 2))
    depth = model.shape[0]
    r_frac = []

    # sep.sum_ellipse is very slow! Try to improve!
    if depth > 1:
        for i in range(depth):
            r_max = max(model.shape)
            r_ = np.linspace(0, r_max, 500)
            flux_ = sep.sum_ellipse(
                model[i], [x_cen], [y_cen], 1, 1 * q[i], pa[i], r=r_)[0]
            flux_ /= total_flux[i]
            func = UnivariateSpline(r_, flux_ - frac, s=0)
            r_frac.append(func.roots()[0])
    else:  # might be buggy
        r_max = max(model.shape)
        r_ = np.linspace(0, r_max, 500)
        flux_ = sep.sum_ellipse(
            model[0], [x_cen], [y_cen], 1, 1 * q[0], pa[0], r=r_)[0]
        flux_ /= total_flux[0]
        func = UnivariateSpline(r_, flux_ - frac, s=0)
        r_frac.append(func.roots()[0])

    return np.array(r_frac) * np.sqrt(q)
