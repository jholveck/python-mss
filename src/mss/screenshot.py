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

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    Buffer = bytes | bytearray | memoryview


class ScreenShot:
    """Screenshot object.

    .. note::

        A better name would have been *Image*, but to prevent collisions
        with PIL.Image, it has been decided to use *ScreenShot*.
    """

    __slots__ = {"__pixels", "__rgb", "_raw", "pos", "size"}

    def __init__(self, data: Buffer, monitor: Monitor, /, *, size: Size | None = None) -> None:
        self.__pixels: Pixels | None = None
        self.__rgb: Buffer | None = None

        # A memoryview holding the raw RGBA pixels retrieved by the OS-specific implementation.  They guarantee it will be valid until released (via the buffer protocol).
        # TODO(jholveck): Should I make this always read-only?
        self._raw: memoryview

        if isinstance(data, memoryview):
            # We currently only get simple 1d memoryviews from _grab_impl.  If that changes (such as to accomodate
            # strided rows, used by some APIs), we'll have to update __buffer__, and review our other uses of raw.
            # We add asserts here to remind us to do that.
            assert data.format == "B"  # noqa: S101
            assert data.strides == (1,)  # noqa: S101
            # I think the next two are redundant with checking that strides is (1,), but I'm making extra-sure.
            assert data.ndim == 1  # noqa: S101
            assert data.c_contiguous  # noqa: S101
            self._raw = data
        else:
            self._raw = memoryview(data)

        #: NamedTuple of the screenshot coordinates.
        self.pos: Pos = Pos(monitor["left"], monitor["top"])

        #: NamedTuple of the screenshot size.
        self.size: Size = Size(monitor["width"], monitor["height"]) if size is None else size

    def __repr__(self) -> str:
        return f"<{type(self).__name__} pos={self.left},{self.top} size={self.width}x{self.height}>"

    def buffer(self, *, nd: bool = True, writable: bool = False) -> memoryview:
        """BGRA values from the BGRA raw pixels.

        This is for users who need extremely fast access to the
        screenshot data, such as for video recording or AI.  It shares
        memory with the internal buffer, and so is the fastest way to
        access the raw data.

        If nd is True (the default), the returned buffer is
        C-contiguous, in HWC layout.  Its elements may be accessed as,
        for instance, buf[0,-1,2] for the red channel of the top-right
        pixel.  If nd is False, the returned buffer will be a contiguous
        1d buffer.  (nd is named for the Python PyBUF_ND flag, meaning
        N-dimensional, as opposed to 1d.)

        If writable is False (the default), the returned buffer is
        read-only.  A writable buffer may be requested, but with some
        caveats: modifications to its contents may not be reflected in
        certain other methods (such as pixels).  Depending on the
        specific platform and backend in use, a writable buffer may not
        be available.

        Starting with Python 3.12, it is not necessary to call this
        method explicitly: the ScreenShot object can act as a buffer
        itself.
        """
        rv = self._raw
        if writable:
            if rv.readonly:
                msg = "This screenshot is read-only, but a writable buffer was requested."
                raise BufferError(msg)
        elif not rv.readonly:
            rv = rv.toreadonly()
        if nd:
            rv = rv.cast("B", [self.size.width, self.size.height, 4])
        return rv

    # The Python-side buffer interface wasn't added until Python 3.12.
    if sys.version_info >= (3, 12):

        def __buffer__(self, flags: int) -> memoryview:
            # We don't check the flags other than WRITABLE and ND, since the other flags are currently always
            # satisfied with the contiguous 1d byte buffers we have in self._raw.  See the comment on self._raw in
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

        .. versionchanged:: 10.3.0
           The returned array is now read-only.  Advanced users who need
           a writable array can request one with the buffer method.
           TODO(jholveck): Should we do this?  NumPy supports read-only
           arrays, but PyTorch doesn't (it will emit a warning and treat
           it read-write).

        .. seealso::

            https://numpy.org/doc/stable/reference/arrays.interface.html
               The NumPy array interface protocol specification
        """
        return {
            "version": 3,
            "shape": (self.height, self.width, 4),
            "typestr": "|u1",
            "data": (self._raw.toreadonly(), True),
        }

    @classmethod
    def from_size(cls: type[ScreenShot], data: Buffer, width: int, height: int, /) -> ScreenShot:
        """Instantiate a new class given only screenshot's data and size."""
        monitor = {"left": 0, "top": 0, "width": width, "height": height}
        return cls(data, monitor)

    @property
    def bgra(self) -> Buffer:
        """BGRx values from the BGRx raw pixels.

        The format is a 1d memoryview of bytes with BGRxBGRx... sequence.
        A specific pixel can be accessed as
        ``bgra[(y * width + x) * 4:(y * width + x) * 4 + 4]``.

        .. version-changed:: 10.3.0
           Prior to this version, this was a bytes object.

        .. note::
            While the name is ``bgra``, the alpha channel may or may not be
            valid.
        """
        return self._raw.toreadonly()

    # TODO(jholveck): Should we broaden the return type to anticipate that we may be able to get multidimensional
    # strided memoryviews or something like that?
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
        if not ((0 <= coord_x < self.size.width) and (0 <= coord_y < self.size_height)):
            msg = f"Pixel location ({coord_x}, {coord_y}) is out of range."
            # Not exactly the usual way to wrap these sorts of errors.
            index_error = IndexError(msg)
            raise ScreenShotError(msg) from index_error
        start_idx = coord_y * self.size.width * 4 + coord_x * 4
        return tuple(reversed(self._raw[start_idx : start_idx + 4]))

    @property
    def rgb(self) -> Buffer:
        """Compute RGB values from the BGRA raw pixels.

        The format is a memoryview object with BGRBGR... sequence.  A
        specific pixel can be accessed as
        ``rgb[(y * width + x) * 3:(y * width + x) * 3 + 3]``.

        .. version-changed:: 10.3.0
           Prior to this version, this was a bytes object.
        """
        if not self.__rgb:
            rgb = bytearray(self.height * self.width * 3)
            raw = self.raw
            rgb[::3] = raw[2::4]
            rgb[1::3] = raw[1::4]
            rgb[2::3] = raw[::4]
            self.__rgb = memoryview(rgb).toreadonly()

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
