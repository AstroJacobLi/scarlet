import autograd.numpy as np
import numpy.ma as ma
import proxmin.operators
from functools import partial

from .bbox import Box, overlapped_slices
from .constraint import (
    ConstraintChain,
    L0Constraint,
    PositivityConstraint,
    PositivityScalesConstraint,
    MonotonicityConstraint,
    MonotonicMaskConstraint,
    SymmetryConstraint,
    CenterOnConstraint,
    NormalizationConstraint,
)
from .frame import Frame
from .model import Model, UpdateException
from .parameter import Parameter, relative_step
from .psf import PSF
from .wavelet import Starlet, starlet_reconstruction, get_multiresolution_support
from . import fft
from . import initialization
from . import profile
import galsim


class Morphology(Model):
    """Morphology base class

    The class describes the 2D image of the spatial dependence of
    `~scarlet.FactorizedComponent`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    parameters: list of `~scarlet.Parameter`
    bbox: `~scarlet.Box`
        2D bounding box of this model
    """

    def __init__(self, frame, *parameters, bbox=None):
        assert isinstance(frame, Frame)
        self.frame = frame

        if bbox is None:
            bbox = frame.bbox
        assert isinstance(bbox, Box)
        self.bbox = bbox

        super().__init__(*parameters)

    def shrink_box(self, image, thresh=0):
        # peel the onion
        size = max(image.shape)
        dist = 0
        while (
            np.all(image[dist, :] <= thresh) and
            np.all(image[-dist - 1, :] <= thresh) and
            np.all(image[:, dist] <= thresh) and
            np.all(image[:, -dist - 1] <= thresh)
        ):
            dist += 1
        newsize = initialization.get_minimal_boxsize(size - 2 * dist)
        if newsize < size:
            dist = (size - newsize) // 2
            # adjust bbox
            self.bbox.origin = tuple(o + dist for o in self.bbox.origin)
            self.bbox.shape = (newsize, newsize)


class ImageMorphology(Morphology):
    """Morphology from an image

    The class uses an arbitrary image as non-parametric model. To allow for subpixel
    offsets, a Fourier-based shifting transformation is available.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    image: 2D array or `~scarlet.Parameter`
        Image parameter
    bbox: `~scarlet.Box`
        2D bounding box for focation of the image in `frame`
    shift: None or `~scarlet.Parameter`
        2D shift parameter (in units of image pixels)
    resizing: bool
        Whether to resize the box dynamically
    """

    def __init__(
        self, frame, image, bbox=None, shifting=False, shift=None, resizing=True, fixed=False
    ):
        if isinstance(image, Parameter):
            assert image.name == "image"
        else:
            constraint = PositivityConstraint()
            image = Parameter(
                image, name="image", step=relative_step, constraint=constraint, fixed=fixed,
            )

        if bbox is None:
            assert frame.bbox[1:].shape == image.shape
            bbox = Box(image.shape)
        else:
            assert bbox.shape == image.shape

        self.resizing = resizing
        self.shifting = shifting

        # create the shift parameter to allow for dynamic resizing
        if shift is None:
            shift = Parameter(np.zeros(2), name="shift", step=1e-2, fixed=self.shifting)
        else:
            assert shift.shape == (2,)
            if isinstance(shift, Parameter):
                assert shift.name == "shift"
            else:
                shift = Parameter(shift, name="shift", step=1e-2)

        parameters = (image, shift)
        super().__init__(frame, *parameters, bbox=bbox)

    def get_model(self, *parameters):
        image = self.get_parameter(0, *parameters)
        shift = self.get_parameter(1, *parameters)

        if self.shifting:
            image = fft.shift(image, shift, return_Fourier=False)
        return image

    def update(self):
        image = self._parameters[0]

        if not self.resizing or image.fixed:
            return

        # shrink the box?
        bbox = self.bbox.copy()
        self.shrink_box(image)
        if bbox != self.bbox:
            slice, _ = overlapped_slices(bbox, self.bbox)
            image = Parameter(
                image[slice],
                name=image.name,
                prior=image.prior,
                constraint=image.constraint,
                step=image.step / 2,
                fixed=image.fixed,
                m=image.m[slice] if image.m is not None else None,
                v=image.v[slice] if image.v is not None else None,
                vhat=image.vhat[slice] if image.vhat is not None else None,
            )

            # set new parameters
            self._parameters = (image,) + self._parameters[1:]

            raise UpdateException

        # grow the box?
        # because the PSF moves power across the box, the gradients at the edge
        # accummulate flux from beyond the box
        elif image.m is not None:
            # next adam gradient update
            gu = -image.m / np.sqrt(np.sqrt(ma.masked_equal(image.v, 0))) * image.step
            gu_pull = gu * (image > 0)  # check if model has flux at the edge at all
            edge_pull = np.array(
                (
                    gu_pull[:, 0].mean(),
                    gu_pull[:, -1].mean(),
                    gu_pull[0, :].mean(),
                    gu_pull[-1, :].mean(),
                )
            )

            # 0.1 compared to 1 at center
            if np.any(edge_pull > 0.1):
                # find next larger boxsize
                size = max(bbox.shape)
                newsize = initialization.get_minimal_boxsize(size + 1)
                pad_width = (newsize - size) // 2

                # Create new parameter for extended image
                image = Parameter(
                    np.pad(image, pad_width, mode="linear_ramp"),
                    name=image.name,
                    prior=image.prior,
                    constraint=image.constraint,
                    step=image.step / 2,
                    fixed=image.fixed,
                    m=np.pad(image.m, pad_width, mode="constant")
                    if image.m is not None
                    else None,
                    v=np.pad(image.v, pad_width, mode="constant")
                    if image.v is not None
                    else None,
                    vhat=np.pad(image.vhat, pad_width, mode="constant")
                    if image.vhat is not None
                    else None,
                )
                # set new parameters
                self._parameters = (image,) + self._parameters[1:]

                # adjust bbox
                self.bbox.origin = tuple(o - pad_width for o in self.bbox.origin)
                self.bbox.shape = (newsize, newsize)
                raise UpdateException


class PointSourceMorphology(Morphology):
    """Morphology from a PSF

    The class uses `frame.psf` as model, evaluated at `center`

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    center: array or `~scarlet.Parameter`
        2D center parameter (in units of frame pixels)
    """

    def __init__(self, frame, center):

        assert frame.psf is not None and isinstance(frame.psf, PSF)
        self.psf = frame.psf

        # define bbox
        pixel_center = tuple(np.round(center).astype("int"))
        shift = (0, *pixel_center)
        bbox = self.psf.bbox + shift

        # parameters is simply 2D center
        if isinstance(center, Parameter):
            assert center.name == "center"
            self.center = center
        else:
            self.center = Parameter(center, name="center", step=3e-2)

        super().__init__(frame, self.center, bbox=bbox)

    def get_model(self, *parameters):
        center = self.get_parameter(0, *parameters)
        box_center = np.mean(self.bbox.bounds[1:], axis=1)
        offset = center - box_center
        return self.psf.get_model(offset=offset)  # no "internal" PSF parameters here


class StarletMorphology(Morphology):
    """Morphology from a starlet representation of an image

    The class uses the starlet parameterization as an overcomplete, non-parametric model.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    image: 2D array
        Initial image to construct starlet transform
    bbox: `~scarlet.Box`
        2D bounding box for focation of the image in `frame`
    monotonic: bool
        Whether to constrain every starlet scale to be monotonic; otherwise they are
        hard-thresholded by `threshold`.
    threshold: float
        Lower bound on threshold for all but the last starlet scale
    """

    def __init__(self, frame, image, bbox=None, monotonic=False,
                 threshold=0, variance=0.05, scales=[0, 1, 2, 3, 4]):

        if bbox is None:
            assert frame.bbox[1:].shape == image.shape
            bbox = Box(image.shape)

        self.monotonic = monotonic
        self.variance = variance
        self.scales = scales
        self.thresh = threshold

        # Starlet transform of morphologies (n1,n2) with 3 dimensions: (scales+1,n1,n2)
        self.transform = Starlet.from_image(image)
        # The starlet transform is the model
        coeffs = self.transform.coefficients

        # wavelet-scale norm
        starlet_norm = self.transform.norm
        # One threshold per wavelet scale: thresh*norm
        thresh_array = np.zeros(coeffs.shape) + self.thresh
        thresh_array *= starlet_norm[:, None, None]
        # We don't threshold the last scale
        thresh_array[-1] = 0

        if not self.monotonic:
            constraint = ConstraintChain(
                PositivityConstraint(0), L0Constraint(thresh_array)
            )
        else:
            # We only apply positive and monotonic constraints to scale 1&2
            # Then we apply the old wavelet constrains on other scales.
            thresh_array[self.scales[-1]:] = 0  # no threshold on large scales
            center = tuple(s // 2 for s in bbox.shape)
            constraint = ConstraintChain(
                MonotonicMaskConstraint(center, variance=self.variance, scales=self.scales, center_radius=1),
                L0Constraint(thresh_array)
            )

        coeffs = Parameter(coeffs, name="coeffs", step=1e-2, constraint=constraint)
        super().__init__(frame, coeffs, bbox=bbox)

    def get_model(self, *parameters):
        # Takes the inverse transform of parameters as starlet coefficients
        coeffs = self.get_parameter(0, *parameters)
        return starlet_reconstruction(coeffs)

    def update(self):
        coeffs = self.get_parameter(0)
        if coeffs.fixed:
            return

        # shrink the box?
        image = self.get_model()
        bbox = self.bbox.copy()
        self.shrink_box(image, thresh=1e-8)
        if bbox != self.bbox:
            slice, _ = overlapped_slices(bbox, self.bbox)
            center = tuple(s // 2 for s in self.bbox.shape)
            coeffs = self.transform.coefficients
            # wavelet-scale norm
            starlet_norm = self.transform.norm
            # One threshold per wavelet scale: thresh*norm
            thresh_array = np.zeros(coeffs.shape) + self.thresh
            thresh_array *= starlet_norm[:, None, None]
            # We don't threshold the last scale
            thresh_array[-1] = 0
            if self.monotonic:
                # non-monotonic can keep its constraint as it's independent of size
                thresh_array[self.scales[-1]:] = 0  # no threshold on large scales
                constraint = ConstraintChain(
                    MonotonicMaskConstraint(center, variance=self.variance,
                                            scales=self.scales, center_radius=1),
                    L0Constraint(thresh_array)
                )
                # constraint = MonotonicMaskConstraint(center, center_radius=1)
            coeffs = Parameter(
                coeffs[:, slice[0], slice[1]],
                name=coeffs.name,
                prior=coeffs.prior,
                constraint=constraint,
                step=coeffs.step,
                fixed=coeffs.fixed,
                m=coeffs.m[:, slice[0], slice[1]] if coeffs.m is not None else None,
                v=coeffs.v[:, slice[0], slice[1]] if coeffs.v is not None else None,
                vhat=coeffs.vhat[:, slice[0], slice[1]]
                if coeffs.vhat is not None
                else None,
            )

            # set new parameters
            self._parameters = (coeffs,) + self._parameters[1:]

            raise UpdateException


class ExtendedSourceMorphology(ImageMorphology):
    def __init__(
        self,
        frame,
        center,
        image,
        bbox=None,
        monotonic="angle",
        symmetric=False,
        min_grad=0,
        shifting=False,
        resizing=True,
    ):
        """Non-parametric image morphology designed for galaxies as extended sources.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the full model
        center: tuple
            Center of the source
        image: `numpy.ndarray`
            Image of the source.
        bbox: `~scarlet.Box`
            2D bounding box for focation of the image in `frame`
        monotonic: ['flat', 'angle', 'nearest'] or None
            Which version of monotonic decrease in flux from the center to enforce
        symmetric: `bool`
            Whether or not to enforce symmetry.
        min_grad: float in [0,1)
            Minimal radial decline for monotonicity (in units of reference pixel value)
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        resize: bool
            Whether to resize the box dynamically
        """

        constraints = []
        # backwards compatibility: monotonic was boolean
        if monotonic is True:
            monotonic = "angle"
        elif monotonic is False:
            monotonic = None
        if monotonic is not None:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(
                MonotonicityConstraint(neighbor_weight=monotonic, min_gradient=min_grad)
            )

        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max"),
        ]
        morph_constraint = ConstraintChain(*constraints)
        image = Parameter(image, name="image", step=1e-2, constraint=morph_constraint)

        self.pixel_center = np.round(center).astype("int")
        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None
        self.shift = shift

        super().__init__(
            frame, image, bbox=bbox, shifting=shifting, shift=shift, resizing=resizing
        )

    @property
    def center(self):
        if self.shift is not None:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center


class SpergelMorphology(Morphology):
    """Morphology from a Spergel profile.

    Start with a round Spergel profile.
    We don't care about flux because it is taken into account by SED.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    center: array or `~scarlet.Parameter`
        2D center parameter (in units of frame pixels)
    """

    def __init__(self,
                 frame,
                 center,
                 nu,
                 rhalf,
                 g1,
                 g2,
                 bbox=None,
                 shifting=False):
        """
        Spergel Morphology

        Parameters
        ----------
        nu, rhalf, g1, g2 are initial guesses for the Spergel parameters.
        """

        # Center of the Spergel profile
        self.pixel_center = np.round(center).astype("int")

        # define bbox
        if bbox is None:
            bbox = frame.bbox  # default to full frame
        else:
            assert len(bbox.shape) == len(frame.bbox.shape)

        # parameters is simply 2D center (i.e., the shift from pixel_center)
        if shifting:
            self.shift = Parameter(center - self.pixel_center, name="shift", step=1e-2)
        else:
            self.shift = Parameter(np.zeros(2) + 0.01, name="shift", step=1e-2, fixed=True)

        # Spergel parameters
        if isinstance(nu, Parameter):
            assert nu.name == "nu"
            self.nu = nu
        else:
            self.nu = Parameter(nu, name="nu", step=1e-2)

        if isinstance(rhalf, Parameter):
            assert rhalf.name == "rhalf"
            self.rhalf = rhalf
        else:
            self.rhalf = Parameter(rhalf, name="rhalf", step=1e-2)

        if isinstance(g1, Parameter):
            assert g1.name == "g1"
            self.g1 = g1
        else:
            self.g1 = Parameter(g1, name="g1", step=1e-2)

        if isinstance(g2, Parameter):
            assert g2.name == "g2"
            self.g2 = g2
        else:
            self.g2 = Parameter(g2, name="g2", step=1e-2)

        # david = Parameter(np.array([nu, rhalf, g1, g2], dtype=np.float32), name="david", step=1e-2)
        # steps of 1% of mean amplitude, minimum set by noise_rms
        # step = partial(relative_step, factor=1e-2, minimum=np.array([1e-2, 1e-2, 1e-2, 1e-2]))
        # david = Parameter(
        #     david, name="david", step=step, constraint=None,  # =PositivityConstraint(zero=1e-20)
        # )
        # parameters = (nu, rhalf, g1, g2, shift)
        self._f = profile.SpergelProfile(boxsize=bbox.shape[1])
        super().__init__(frame, self.nu, self.rhalf, self.g1, self.g2, self.shift, bbox=bbox)

    def center(self):
        if self.shift is not None:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center

    def get_model(self, *parameters, shift=None):
        if shift is None:
            shift = self.shift

        return self._f.get_model(*parameters, shift=shift)
        # nu = self.get_parameter(0, *parameters)  # Spergel parameters
        # rhalf = self.get_parameter(1, *parameters)
        # g1 = self.get_parameter(2, *parameters)
        # g2 = self.get_parameter(3, *parameters)
        # shift = self.get_parameter(4, *parameters)
        return profile.SpergelProfile(boxsize=self.bbox.shape[1]).get_model(*parameters)
