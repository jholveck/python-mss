# This is part of the MSS Python's module.
# Source: https://github.com/BoboTiG/python-mss.

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

from mss.exception import ScreenShotError
from mss.models import Monitor, Pixel, Pixels, Pos, Size

if TYPE_CHECKING:  # pragma: nocover
    from collections.abc import Iterator


class ScreenShot:
    """Screenshot object.

    .. note::

        A better name would have  been *Image*, but to prevent collisions
        with PIL.Image, it has been decided to use *ScreenShot*.
    """

    __slots__ = {"__bgra", "__pixels", "__rgb", "pos", "raw", "size"}

    def __init__(self, data: bytes | bytearray | memoryview, monitor: Monitor, /, *, size: Size | None = None) -> None:
        self.__pixels: Pixels | None = None
        self.__rgb: bytes | None = None
        self.__bgra: bytes | None = None

        #: Bytes-like object (bytes, bytearray, or memoryview) of the raw BGRA pixels retrieved by ctypes
        #: OS independent implementations.
        self.raw: bytearray = data

        if isinstance(data, memoryview):
            # We currently only get simple 1d memoryviews from _grab_impl.  If that changes (such as to accomodate
            # strided rows), we'll have to update __buffer__, and review our other uses of raw.  We add asserts here
            # to remind us to do that.
            assert data.format == "B"  # noqa: S101
            assert data.strides == (1,)  # noqa: S101
            # I think the next two are redundant with checking that strides is (1,), but I'm making extra-sure.
            assert data.ndim == 1  # noqa: S101
            assert data.c_contiguous  # noqa: S101

        #: NamedTuple of the screenshot coordinates.
        self.pos: Pos = Pos(monitor["left"], monitor["top"])

        #: NamedTuple of the screenshot size.
        self.size: Size = Size(monitor["width"], monitor["height"]) if size is None else size

        # Pixel access would benefit from caching a dimensional memoryview (like the one we get from buffer) like this
        # around, for things like fast indexed pixel access and .tolist.  However, right now it's not helpful because
        # of the current API, which (for instance) uses tuples for RGBA, preventing .tolist.  It's a minor change, but
        # incompatible.
        # TODO(jholveck): At the next major update, consider changing the relevant APIs.

    def __repr__(self) -> str:
        return f"<{type(self).__name__} pos={self.left},{self.top} size={self.width}x{self.height}>"

    def buffer(self, *, nd: bool = True, writable: bool = False) -> memoryview:
        """BGRA values from the BGRA raw pixels.

        This is for users who need extremely fast access to the
        screenshot data, such as for video recording or AI.  It shares
        memory with the internal buffer, and so is the fastest way to
        access the raw data.

        By default, the returned buffer is C-contiguous, in HWC layout.
        Its elements may be accessed as, for instance, buf[0,-1,2] for
        the red channel of the top-right pixel.  If nd (so-named for
        the Python PyBUF_ND flag) is False, the returned buffer will be
        a contiguous 1d array.

        By default, the returned buffer is read-only.  A writable buffer
        may be requested, but is for *advanced users only*.  Modifying
        its contents may cause some other methods (such as pixels) to
        behave incorrectly.  Depending on the specific platform and
        backend in use, a writable buffer may not be available.
        """
        rv = self.raw if isinstance(self.raw, memoryview) else memoryview(self.raw)
        if writable:
            if rv.readonly:
                msg = "This screenshot is read-only, but a writable buffer was requested."
                raise TypeError(msg)
        elif not rv.readonly:
            rv = rv.toreadonly()
        if nd:
            rv = rv.cast("B", [self.size.width, self.size.height, 4])
        return rv

    # The Python-side buffer interface wasn't added until Python 3.12.
    if sys.version_info >= (3, 12):

        def __buffer__(self, flags: int) -> memoryview:
            # We don't check the flags other than WRITABLE and ND, since the other flags are currently always
            # satisfied with the contiguous 1d byte buffers we have in self.raw.  See the comment on self.raw in
            # __init__.
            rv = self.buffer(nd=flags & inspect.BufferFlags.ND, writable=bool(flags & inspect.BufferFlags.WRITABLE))
            return rv.cast("B")

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """NumPy array interface support.

        This is used by NumPy, many SciPy projects, CuPy, PyTorch (via
        ``torch.from_numpy``), TensorFlow (via ``tf.convert_to_tensor``),
        JAX (via ``jax.numpy.asarray``), Pandas, scikit-learn, Matplotlib,
        some OpenCV functions, and others.  This allows you to pass a
        :class:`ScreenShot` instance directly to these libraries without
        needing to convert it first.

        This is in HWC order, with 4 channels (BGRA).

        .. seealso::

            https://numpy.org/doc/stable/reference/arrays.interface.html
               The NumPy array interface protocol specification
        """
        # TODO(jholveck): In the next major release, we should return a read-only buffer.  That's because we cache
        # data in __rgb and __pixels, so we don't want the user to change it.  Otherwise, there may be unpredictable
        # results.  (Advanced users who know the consequences could still ask for a read-write buffer and pass that to
        # NumPy.)  However, this would be a mildly backwards-incompatible change; some users might be already doing
        # some in-place image manipulation.
        return {
            "version": 3,
            "shape": (self.height, self.width, 4),
            "typestr": "|u1",
            "data": self.raw,
        }

    @classmethod
    def from_size(cls: type[ScreenShot], data: bytearray, width: int, height: int, /) -> ScreenShot:
        """Instantiate a new class given only screenshot's data and size."""
        monitor = {"left": 0, "top": 0, "width": width, "height": height}
        return cls(data, monitor)

    @property
    def bgra(self) -> bytes:
        """BGRx values from the BGRx raw pixels.

        The format is a bytes object with BGRxBGRx... sequence.  A specific
        pixel can be accessed as
        ``bgra[(y * width + x) * 4:(y * width + x) * 4 + 4].``

        .. note::
            While the name is ``bgra``, the alpha channel may or may not be
            valid.
        """
        return bytes(self.raw)

    @property
    def pixels(self) -> Pixels:
        """RGB tuples.

        The format is a list of rows.  Each row is a list of pixels.
        Each pixel is a tuple of (R, G, B).
        """
        if not self.__pixels:
            rgb_tuples: Iterator[Pixel] = zip(self.raw[2::4], self.raw[1::4], self.raw[::4])
            self.__pixels = list(zip(*[iter(rgb_tuples)] * self.width))

        return self.__pixels

    def pixel(self, coord_x: int, coord_y: int) -> Pixel:
        """Return the pixel value at a given position.

        :returns: A tuple of (R, G, B) values.
        """
        try:
            return self.pixels[coord_y][coord_x]
        except IndexError as exc:
            msg = f"Pixel location ({coord_x}, {coord_y}) is out of range."
            raise ScreenShotError(msg) from exc

    @property
    def rgb(self) -> bytes:
        """Compute RGB values from the BGRA raw pixels.

        The format is a bytes object with BGRBGR... sequence.  A specific
        pixel can be accessed as
        ``rgb[(y * width + x) * 3:(y * width + x) * 3 + 3]``.
        """
        if not self.__rgb:
            rgb = bytearray(self.height * self.width * 3)
            raw = self.raw
            rgb[::3] = raw[2::4]
            rgb[1::3] = raw[1::4]
            rgb[2::3] = raw[::4]
            self.__rgb = bytes(rgb)

        return self.__rgb

    @property
    def top(self) -> int:
        """Convenient accessor to the top position."""
        return self.pos.top

    @property
    def left(self) -> int:
        """Convenient accessor to the left position."""
        return self.pos.left

    @property
    def width(self) -> int:
        """Convenient accessor to the width size."""
        return self.size.width

    @property
    def height(self) -> int:
        """Convenient accessor to the height size."""
        return self.size.height
