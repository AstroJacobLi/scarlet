import autograd.numpy as np
import autograd.scipy as scipy
from .bbox import Box
from .model import Model, abstractmethod
from .parameter import Parameter
from .fft import Fourier, shift
from scipy.special import kv, gamma
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def normalize(image):
    """Normalize to PSF image in every band to unity
    """
    sums = np.nansum(image, axis=(1, 2))
    if isinstance(image, Parameter):
        image._data /= sums[:, None, None]
    else:
        image /= sums[:, None, None]
    return image


class Profile(Model):
    @abstractmethod
    def get_model(self, *parameter, offset=None):
        """Get the Profile realization

        Parameters
        ----------
        parameters: tuple of optimimzation parameters
        offset: 2D tuple or ``~scarlet.Parameter`
            Optional subpixel offset of the model, in units of frame pixels

        Returns
        -------
        result: array
            A centered Profile model defined by its parameters, shifted by `offset`
        """
        pass

    def prepare_param(self, X, name):
        if isinstance(X, Parameter):
            assert X.name == name
        else:
            if np.isscalar(X):
                X = (X,)
            X = Parameter(np.array(X, dtype="float"), name=name, fixed=False)
        return X


class FunctionProfile(Profile):
    """Base class for Surface Brightnee Profiles with functional forms.

    Parameters
    ----------
    parameters: `~scarlet.Parameter` or list thereof
        Optimization parameters. Can be fixed.
    integrate: bool
        Whether pixel integration is performed
    boxsize: int
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, *parameters, integrate=True, boxsize=None):

        super().__init__(*parameters)

        self.integrate = integrate

        if boxsize is None:
            boxsize = 200
        if boxsize % 2 == 0:
            boxsize += 1

        # length of 0 parameter gives number of channels
        p0 = self.get_parameter(0, *parameters)
        shape = (len(p0), boxsize, boxsize)
        origin = (0, -(boxsize // 2), -(boxsize // 2))
        self.bbox = Box(shape, origin=origin)

        self._Y = np.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        self._X = np.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        # same across all bands
        self.is_same = np.all(p0 == p0[0])
        self._d = self.bbox.D - 2

    def expand_dims(self, model):
        return np.expand_dims(model, axis=tuple(range(self._d)))


class GaussianProfile(FunctionProfile):
    """Circular Gaussian Function

    Parameters
    ----------
    sigma: float, array, or `~scarlet.Parameter`
        Standard deviation of the Gaussian in frame pixels
        If the width is to be optimized, provide a full defined `Parameter`.
    integrate: bool
        Whether pixel integration is performed
    boxsize: int
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, sigma, integrate=True, boxsize=None):

        sigma = self.prepare_param(sigma, "sigma")

        if boxsize is None:
            boxsize = int(np.ceil(10 * np.max(sigma)))

        super().__init__(sigma, integrate=integrate, boxsize=boxsize)

    def get_model(self, *parameters, offset=None):

        sigma = self.get_parameter(0, *parameters)

        if offset is None:
            offset = (0, 0)

        if self.is_same:
            s = sigma[0]
            psfs = self.expand_dims(
                self._f(self._Y - offset[0], s)[:, None] *
                self._f(self._X - offset[1], s)[None, :]
            )
        else:
            psfs = np.stack(
                (
                    self._f(self._Y - offset[0], s)[:, None] *
                    self._f(self._X - offset[1], s)[None, :]
                    for s in sigma
                ),
                axis=0,
            )
        # use image integration instead of analytic for consistency with other PSFs
        return normalize(psfs)

    def _f(self, X, sigma):
        '''The forward model'''
        if not self.integrate:
            return np.exp(-(X ** 2) / (2 * sigma ** 2))
        else:
            sqrt2 = np.sqrt(2)
            return (
                np.sqrt(np.pi / 2) *
                sigma *
                (
                    1 -
                    scipy.special.erfc((0.5 - X) / (sqrt2 * sigma)) +
                    1 -
                    scipy.special.erfc((2 * X + 1) / (2 * sqrt2 * sigma))
                )
            )


class SpergelProfile(FunctionProfile):
    """
    Spergel profile
    """
    _minimum_nu = -0.85
    _maximum_nu = 4.0

    def __init__(self, david, integrate=True, boxsize=None):
        # David is the Spergel parameters: nu, rhalf, g1, g2.
        david = self.prepare_param(david, "david")

        if david[0] < SpergelProfile._minimum_nu:
            raise ValueError("Requested Spergel index is too small, should be in range [{}, {}]".format(
                SpergelProfile._minimum_nu, SpergelProfile._maximum_nu))
        if david[0] > SpergelProfile._maximum_nu:
            raise ValueError("Requested Spergel index is too large, should be in range [{}, {}]".format(
                SpergelProfile._minimum_nu, SpergelProfile._maximum_nu))

        if boxsize is None:
            boxsize = int(np.ceil(10 * np.max(david[1])))

        # Calculate the coeff c_nu in Spergel profile.
        self._calc_cnu()

        super().__init__(david, integrate=integrate, boxsize=boxsize)

    def get_model(self, *parameter, offset=None):
        """
        Get the Spergel profile realization.
        Written based on astropy.modeling.models.Sersic2D.
        """
        david = self.get_parameter(0, *parameter)
        nu, rhalf, g1, g2 = david

        if offset is None:
            offset = (0.01, 0.01)  # avoid singularity at origin

        cnu = self._cnu(nu)
        g = np.sqrt(g1**2 + g2**2)
        shear_matrix = np.linalg.inv(
            np.array([[1 + g1, g2],
                      [g2, 1 - g1]]) / np.sqrt(1 - g**2))

        x, y = np.meshgrid(self._X - offset[0], self._Y - offset[1])
        x_ = shear_matrix[0, 0] * x + shear_matrix[0, 1] * y
        y_ = shear_matrix[1, 0] * x + shear_matrix[1, 1] * y
        z = np.sqrt((x_ / rhalf) ** 2 + (y_ / rhalf) ** 2)

        # amplitude = 1
        model = self.expand_dims(self._f_nu(cnu * z, nu))
        # model = np.nan_to_num(model, nan=0, posinf=0, neginf=0)
        return normalize(model)

    def _f_nu(self, x, nu):
        """
        Eqn 3 in Spergel (2010).
        kv is the modified Bessel function of the second kind.
        gamma is the gamma function.
        """
        return (x / 2)**nu * kv(nu, x) / gamma(nu + 1)

    def _calc_cnu(self):
        """
        Calculate the c_nu parameter used in Spergel profile.
        See Table 1 in Spergel (2010) and the sentence below Eqn (8).
        """
        _nu = np.linspace(SpergelProfile._minimum_nu,
                          SpergelProfile._maximum_nu,
                          1000)

        def func(x, nu):
            return (1 + nu) * self._f_nu(x, nu + 1) - 0.25

        _cnu = [fsolve(lambda x: func(x, v), x0=0.2)[0] for v in _nu]
        self._cnu = interp1d(_nu, _cnu)
        return
