'''
A place for Image[] and related functions.

Note that you need scikit-image installed in order for this module to work.

This module is part of the Mathics/iMathics branch, since the regular Mathics
notebook seems to lack the functionality to inject <img> tags from the kernel
into the notebook interface (yielding an error 'Unknown node type: img').
Jupyter does not have this limitation though.
'''

from mathics.builtin.base import (
    Builtin, Test, BoxConstruct, String)
from mathics.core.expression import (
    Atom, Expression, Integer, Rational, Real, Symbol, from_python)

import six
import base64

try:
    import skimage
    import skimage.io
    import skimage.transform
    import skimage.filters
    import skimage.exposure
    import skimage.feature
    import skimage.filters.rank
    import skimage.morphology
    import skimage.measure

    import PIL
    import PIL.ImageEnhance
    import PIL.ImageOps
    import PIL.ImageFilter

    import numpy

    import matplotlib.cm

    _enabled = True
except ImportError:
    _enabled = False

from io import BytesIO

if _enabled:
    _color_space_conversions = {
        'RGB2Grayscale': skimage.color.rgb2gray,
        'Grayscale2RGB': skimage.color.gray2rgb,

        'HSV2RGB': skimage.color.hsv2rgb,
        'RGB2HSV': skimage.color.rgb2hsv,

        'LAB2LCH': skimage.color.lab2lch,
        'LCH2LAB': skimage.color.lch2lab,

        'LAB2RGB': skimage.color.lab2rgb,
        'LAB2XYZ': skimage.color.lab2xyz,

        'LUV2RGB': skimage.color.luv2rgb,
        'LUV2XYZ': skimage.color.luv2xyz,

        'RGB2LAB': skimage.color.rgb2lab,
        'RGB2LUV': skimage.color.rgb2luv,
        'RGB2XYZ': skimage.color.rgb2xyz,

        'XYZ2LAB': skimage.color.xyz2lab,
        'XYZ2LUV': skimage.color.xyz2luv,
        'XYZ2RGB': skimage.color.xyz2rgb,
    }

# import and export

class ImageImport(Builtin):
    messages = {
        'noskimage': 'image import needs scikit-image in order to work.'
    }

    def apply(self, path, evaluation):
        '''ImageImport[path_?StringQ]'''
        if not _enabled:
            return evaluation.message('ImageImport', 'noskimage')
        else:
            pixels = skimage.io.imread(path.get_string_value())
            is_rgb = len(pixels.shape) >= 3 and pixels.shape[2] >= 3
            atom = Image(pixels, 'RGB' if is_rgb else 'Grayscale')
            return Expression('List', Expression('Rule', String('Image'), atom))


class ImageExport(Builtin):
    messages = {
        'noskimage': 'image export needs scikit-image in order to work.',
        'noimage': 'only an Image[] can be exported into an image file'
    }

    def apply(self, path, expr, opts, evaluation):
        '''ImageExport[path_?StringQ, expr_, opts___]'''
        if not _enabled:
            return evaluation.message('ImageExport', 'noskimage')
        elif isinstance(expr, Image):
            skimage.io.imsave(path.get_string_value(), expr.pixels)
            return Symbol('Null')
        else:
            return evaluation.message('ImageExport', 'noimage')

# image math


class _ImageArithmetic(Builtin):
    ufunc = None    # must be implemented

    messages = {
        'bddarg': 'Expecting a number, image, or graphics instead of `1`.',
    }

    @staticmethod
    def convert_Image(image):
        assert isinstance(image, Image)
        return skimage.img_as_float(image.pixels)

    @staticmethod
    def convert_args(*args):
        images = []
        for arg in args:
            if isinstance(arg, Image):
                images.append(_ImageArithmetic.convert_Image(arg))
            elif isinstance(arg, (Integer, Rational, Real)):
                images.append(float(arg.to_python()))
            else:
                return None, arg
        return images, None

    @staticmethod
    def _reduce(iterable, ufunc):
        result = None
        for i in iterable:
            if result is None:
                # ufunc is destructive so copy first
                result = numpy.copy(i)
            else:
                # e.g. result *= i
                ufunc(result, i, result)
        return result

    def apply(self, image, args, evaluation):
        '%(name)s[image_Image, args__]'
        images, arg = self.convert_args(image, *args.get_sequence())
        if images is None:
            return evaluation.message(self.get_name(), 'bddarg', arg)
        result = self._reduce(images, self.ufunc)
        return Image(result.clip(0, 1), image.color_space)


class ImageAdd(_ImageArithmetic):
    '''
    <dl>
    <dt>'ImageAdd[$image$, $expr_1$, $expr_2$, ...]'
      <dd>adds all $expr_i$ to $image$ where each $expr_i$ must be an image or a real number.
    </dl>

    >> i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}];

    >> ImageAdd[i, 0.5]
     = -Image-

    >> ImageAdd[i, i]
     = -Image-

    #> ImageAdd[i, 0.2, i, 0.1]
     = -Image-

    #> ImageAdd[i, x]
     : Expecting a number, image, or graphics instead of x.
     = ImageAdd[-Image-, x]
    '''
    ufunc = numpy.add


class ImageSubtract(_ImageArithmetic):
    '''
    <dl>
    <dt>'ImageSubtract[$image$, $expr_1$, $expr_2$, ...]'
      <dd>subtracts all $expr_i$ from $image$ where each $expr_i$ must be an image or a real number.
    </dl>

    >> i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}];

    >> ImageSubtract[i, 0.2]
     = -Image-

    >> ImageSubtract[i, i]
     = -Image-

    #> ImageSubtract[i, 0.2, i, 0.1]
     = -Image-

    #> ImageSubtract[i, x]
     : Expecting a number, image, or graphics instead of x.
     = ImageSubtract[-Image-, x]
    '''
    ufunc = numpy.subtract


class ImageMultiply(_ImageArithmetic):
    '''
    <dl>
    <dt>'ImageMultiply[$image$, $expr_1$, $expr_2$, ...]'
      <dd>multiplies all $expr_i$ with $image$ where each $expr_i$ must be an image or a real number.
    </dl>

    >> i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}];

    >> ImageMultiply[i, 0.2]
     = -Image-

    >> ImageMultiply[i, i]
     = -Image-

    #> ImageMultiply[i, 0.2, i, 0.1]
     = -Image-

    #> ImageMultiply[i, x]
     : Expecting a number, image, or graphics instead of x.
     = ImageMultiply[-Image-, x]
    '''
    ufunc = numpy.multiply


class RandomImage(Builtin):
    rules = {
        'RandomImage[max_?RealNumberQ, {w_Integer, h_Integer}]': 'RandomImage[{0, max}, {w, h}]'
    }

    def apply(self, minval, maxval, w, h, evaluation):
        'RandomImage[{minval_?RealNumberQ, maxval_?RealNumberQ}, {w_Integer, h_Integer}]'
        try:
            x0 = max(minval.to_python(), 0)
            x1 = min(maxval.to_python(), 1)
            return Image((numpy.random.rand(h.to_python(), w.to_python()) * (x1 - x0) + x0), 'Grayscale')
        except:
            import sys
            return String(repr(sys.exc_info()))

# simple image manipulation

class ImageResize(Builtin):
    options = {
        'Resampling': '"Bicubic"'
    }

    messages = {
        'resamplingerr': 'Resampling mode `` is not supported.',
        'gaussaspect': 'Gaussian resampling needs to main aspect ratio.'
    }

    def apply_resize_width(self, image, width, evaluation, options):
        'ImageResize[image_Image, width_Integer, OptionsPattern[ImageResize]]'
        shape = image.pixels.shape
        height = int((float(shape[0]) / float(shape[1])) * width.to_python())
        return self.apply_resize_width_height(image, width, Integer(height), evaluation, options)

    def apply_resize_width_height(self, image, width, height, evaluation, options):
        'ImageResize[image_Image, {width_Integer, height_Integer}, OptionsPattern[ImageResize]]'
        resampling = self.get_option(options, 'Resampling', evaluation)
        resampling_name = resampling.get_string_value() if isinstance(resampling, String) else resampling.to_python()

        w = int(width.to_python())
        h = int(height.to_python())
        if resampling_name == 'Nearest':
            pixels = skimage.transform.resize(image.pixels, (h, w), order=0)
        elif resampling_name == 'Bicubic':
            pixels = skimage.transform.resize(image.pixels, (h, w), order=3)
        elif resampling_name == 'Gaussian':
            old_shape = image.pixels.shape
            sy = h / old_shape[0]
            sx = w / old_shape[1]
            if sy > sx:
                err = abs((sy * old_shape[1]) - (sx * old_shape[1]))
                s = sy
            else:
                err = abs((sy * old_shape[0]) - (sx * old_shape[0]))
                s = sx
            if err > 1.5:
                return evaluation.error('ImageResize', 'gaussaspect')
            elif s > 1:
                pixels = skimage.transform.pyramid_expand(image.pixels, upscale=s).clip(0, 1)
            else:
                pixels = skimage.transform.pyramid_reduce(image.pixels, downscale=1 / s).clip(0, 1)
        else:
            return evaluation.error('ImageResize', 'resamplingerr', resampling_name)

        return Image(pixels, image.color_space)


class ImageReflect(Builtin):
    def apply(self, image, evaluation):
        'ImageReflect[image_Image]'
        return Image(numpy.flipud(image.pixels), image.color_space)

    def apply_ud(self, image, evaluation):
        'ImageReflect[image_Image, Top|Bottom]'
        return Image(numpy.flipud(image.pixels), image.color_space)

    def apply_lr(self, image, evaluation):
        'ImageReflect[image_Image, Left|Right]'
        return Image(numpy.fliplr(image.pixels), image.color_space)


class ImageRotate(Builtin):
    rules = {
        'ImageRotate[i_Image]': 'ImageRotate[i, 90]'
    }

    def apply(self, image, angle, evaluation):
        'ImageRotate[image_Image, angle_?RealNumberQ]'
        return Image(skimage.transform.rotate(image.pixels, angle.to_python(), resize=True), image.color_space)


class ImagePartition(Builtin):
    rules = {
        'ImagePartition[i_Image, s_Integer]': 'ImagePartition[i, {s, s}]'
    }

    def apply(self, image, w, h, evaluation):
        'ImagePartition[image_Image, {w_Integer, h_Integer}]'
        w = w.to_python()
        h = h.to_python()
        pixels = image.pixels
        shape = pixels.shape
        parts = [Image(pixels[y:y + w, x:x + w], image.color_space)
                 for x in range(0, shape[1], w) for y in range(0, shape[0], h)]
        return Expression('List', *parts)

# simple image filters

class ImageAdjust(Builtin):
    def apply_auto(self, image, evaluation):
        'ImageAdjust[image_Image]'
        pixels = skimage.img_as_ubyte(image.pixels)
        return Image(numpy.array(PIL.ImageOps.equalize(PIL.Image.fromarray(pixels))), image.color_space)

    def apply_contrast(self, image, c, evaluation):
        'ImageAdjust[image_Image, c_?RealNumberQ]'
        enhancer_c = PIL.ImageEnhance.Contrast(image.pil())
        return Image(numpy.array(enhancer_c.enhance(c.to_python())), image.color_space)

    def apply_contrast_brightness(self, image, c, b, evaluation):
        'ImageAdjust[image_Image, {c_?RealNumberQ, b_?RealNumberQ}]'
        im = image.pil()
        enhancer_b = PIL.ImageEnhance.Brightness(im)
        im = enhancer_b.enhance(b.to_python())  # brightness first!
        enhancer_c = PIL.ImageEnhance.Contrast(im)
        return Image(numpy.array(enhancer_c.enhance(c.to_python())), image.color_space)


class Blur(Builtin):
    rules = {
        'Blur[i_Image]': 'Blur[i, 2]'
    }

    def apply(self, image, r, evaluation):
        'Blur[image_Image, r_?RealNumberQ]'
        return Image(numpy.array(image.pil().filter(
            PIL.ImageFilter.GaussianBlur(r.to_python()))), image.color_space)


class Sharpen(Builtin):
    rules = {
        'Sharpen[i_Image]': 'Sharpen[i, 2]'
    }

    def apply(self, image, r, evaluation):
        'Sharpen[image_Image, r_?RealNumberQ]'
        return Image(numpy.array(image.pil().filter(
            PIL.ImageFilter.UnsharpMask(r.to_python()))), image.color_space)


class GaussianFilter(Builtin):
    messages = {
        'only3': 'GaussianFilter only supports up to three channels.'
    }

    def apply_radius(self, image, radius, evaluation):
        'GaussianFilter[image_Image, radius_?RealNumberQ]'
        if len(image.pixels.shape) > 2 and image.pixels.shape[2] > 3:
            return evaluation.message('GaussianFilter', 'only3')
        else:
            return Image(skimage.filters.gaussian(
                skimage.img_as_float(image.pixels),
                sigma=radius.to_python() / 2, multichannel=True), image.color_space)

# morphological image filters

class PillowImageFilter(Builtin):
    def compute(self, image, f):
        return Image(numpy.array(image.pil().filter(f)), image.color_space)


class MinFilter(PillowImageFilter):
    def apply(self, image, r, evaluation):
        'MinFilter[image_Image, r_Integer]'
        return self.compute(image, PIL.ImageFilter.MinFilter(1 + 2 * r.to_python()))


class MaxFilter(PillowImageFilter):
    def apply(self, image, r, evaluation):
        'MaxFilter[image_Image, r_Integer]'
        return self.compute(image, PIL.ImageFilter.MaxFilter(1 + 2 * r.to_python()))


class MedianFilter(PillowImageFilter):
    def apply(self, image, r, evaluation):
        'MedianFilter[image_Image, r_Integer]'
        return self.compute(image, PIL.ImageFilter.MedianFilter(1 + 2 * r.to_python()))


class EdgeDetect(Builtin):
    rules = {
        'EdgeDetect[i_Image]': 'EdgeDetect[i, 2, 0.2]',
        'EdgeDetect[i_Image, r_?RealNumberQ]': 'EdgeDetect[i, r, 0.2]'
    }

    def apply(self, image, r, t, evaluation):
        'EdgeDetect[image_Image, r_?RealNumberQ, t_?RealNumberQ]'
        return Image(skimage.feature.canny(
            image.grayscale().pixels, sigma=r.to_python() / 2,
            low_threshold=0.5 * t.to_python(), high_threshold=t.to_python()),
            'Grayscale')


class BoxMatrix(Builtin):
    def apply(self, r, evaluation):
        'BoxMatrix[r_?RealNumberQ]'
        s = 1 + 2 * r.to_python()
        return from_python(skimage.morphology.rectangle(s, s).tolist())


class DiskMatrix(Builtin):
    def apply(self, r, evaluation):
        'DiskMatrix[r_?RealNumberQ]'
        return from_python(skimage.morphology.disk(r).tolist())


class DiamondMatrix(Builtin):
    def apply(self, r, evaluation):
        'DiamondMatrix[r_?RealNumberQ]'
        return from_python(skimage.morphology.diamond(r).tolist())


class MorphologyFilter(Builtin):
    messages = {
        'grayscale': 'Your image has been converted to grayscale as color images are not supported yet.'
    }

    def compute(self, image, f, k, evaluation):
        if image.color_space != 'Grayscale':
            image = image.color_convert('Grayscale')
            evaluation.message('MorphologyFilter', 'grayscale')
        return Image(f(image.pixels, numpy.array(k.to_python())), 'Grayscale')


class Dilation(MorphologyFilter):
    rules = {
        'Dilation[i_Image, r_?RealNumberQ]': 'Dilation[i, BoxMatrix[r]]'
    }

    def apply(self, image, k, evaluation):
        'Dilation[image_Image, k_?MatrixQ]'
        return self.compute(image, skimage.morphology.dilation, k, evaluation)


class Erosion(MorphologyFilter):
    rules = {
        'Erosion[i_Image, r_?RealNumberQ]': 'Erosion[i, BoxMatrix[r]]'
    }

    def apply(self, image, k, evaluation):
        'Erosion[image_Image, k_?MatrixQ]'
        return self.compute(image, skimage.morphology.erosion, k, evaluation)


class Opening(MorphologyFilter):
    rules = {
        'Opening[i_Image, r_?RealNumberQ]': 'Opening[i, BoxMatrix[r]]'
    }

    def apply(self, image, k, evaluation):
        'Opening[image_Image, k_?MatrixQ]'
        return self.compute(image, skimage.morphology.opening, k, evaluation)


class Closing(MorphologyFilter):
    rules = {
        'Closing[i_Image, r_?RealNumberQ]': 'Closing[i, BoxMatrix[r]]'
    }

    def apply(self, image, k, evaluation):
        'Closing[image_Image, k_?MatrixQ]'
        return self.compute(image, skimage.morphology.closing, k, evaluation)


class MorphologicalComponents(Builtin):
    rules = {
        'MorphologicalComponents[i_Image]': 'MorphologicalComponents[i, 0]'
    }

    def apply(self, image, t, evaluation):
        'MorphologicalComponents[image_Image, t_?RealNumberQ]'
        pixels = skimage.img_as_ubyte(skimage.img_as_float(image.grayscale().pixels) > t.to_python())
        return from_python(skimage.measure.label(pixels, background=0, connectivity=2).tolist())

# color space

class ImageColorSpace(Builtin):
    def apply(self, image, evaluation):
        'ImageColorSpace[image_Image]'
        return String(image.color_space)


class ColorConvert(Builtin):
    def apply(self, image, colorspace, evaluation):
        'ColorConvert[image_Image, colorspace_String]'
        return image.color_convert(colorspace.get_string_value())


class ColorQuantize(Builtin):
    def apply(self, image, n, evaluation):
        'ColorQuantize[image_Image, n_Integer]'
        pixels = skimage.img_as_ubyte(image.color_convert('RGB').pixels)
        im = PIL.Image.fromarray(pixels).quantize(n.to_python())
        im = im.convert('RGB')
        return Image(numpy.array(im), 'RGB')


class Threshold(Builtin):
    options = {
        'Method': '"Cluster"'
    }

    messages = {
        'illegalmethod': 'Method `` is not supported.'
    }

    def apply(self, image, evaluation, options):
        'Threshold[image_Image, OptionsPattern[Threshold]]'
        pixels = image.grayscale().pixels

        method = self.get_option(options, 'Method', evaluation)
        method_name = method.get_string_value() if isinstance(method, String) else method.to_python()
        if method_name == 'Cluster':
            threshold = skimage.filters.threshold_otsu(pixels)
        elif method_name == 'Median':
            threshold = numpy.median(pixels)
        elif method_name == 'Mean':
            threshold = numpy.mean(pixels)
        else:
           return evaluation.error('Threshold', 'illegalmethod', method)

        return Real(threshold)


class Binarize(Builtin):
    def apply(self, image, evaluation):
        'Binarize[image_Image]'
        image = image.grayscale()
        threshold = Expression('Threshold', image).evaluate(evaluation).to_python()
        return Image(image.pixels > threshold, 'Grayscale')

    def apply_t(self, image, t, evaluation):
        'Binarize[image_Image, t_?RealNumberQ]'
        pixels = image.grayscale().pixels
        return Image(pixels > t.to_python(), 'Grayscale')

    def apply_t1_t2(self, image, t1, t2, evaluation):
        'Binarize[image_Image, {t1_?RealNumberQ, t2_?RealNumberQ}]'
        pixels = image.grayscale().pixels
        mask1 = pixels > t1.to_python()
        mask2 = pixels < t2.to_python()
        return Image(mask1 * mask2, 'Grayscale')


class ColorNegate(Builtin):
    def apply(self, image, evaluation):
        'ColorNegate[image_Image]'
        pixels = image.pixels
        anchor = numpy.ndarray(pixels.shape, dtype=pixels.dtype)
        anchor.fill(skimage.dtype_limits(pixels)[1])
        return Image(anchor - pixels, image.color_space)


class ColorSeparate(Builtin):
    def apply(self, image, evaluation):
        'ColorSeparate[image_Image]'
        images = []
        pixels = image.pixels
        if len(pixels.shape) < 3:
            images.append(pixels)
        else:
            for i in range(pixels.shape[2]):
                images.append(Image(pixels[:, :, i], 'Grayscale'))
        return Expression('List', *images)


class Colorize(Builtin):
    messages = {
        'toomany': 'Too many levels.'
    }

    def apply(self, a, evaluation):
        'Colorize[a_?MatrixQ]'

        a = numpy.array(a.to_python())
        n = int(numpy.max(a)) + 1
        if n > 8192:
            return evaluation.error('Colorize', 'toomany')

        cmap = matplotlib.cm.get_cmap('hot', n)
        p = numpy.transpose(numpy.array([cmap(i) for i in range(n)])[:, 0:3])
        s = (a.shape[0], a.shape[1], 1)
        return Image(numpy.concatenate([p[i][a].reshape(s) for i in range(3)], axis=2), color_space='RGB')

# pixel access

class ImageData(Builtin):
    rules = {
        'ImageData[image_Image]': 'ImageData[image, "Real"]'
    }

    messages = {
        'pixelfmt': 'unsupported pixel format "``"'
    }

    def apply(self, image, stype, evaluation):
        'ImageData[image_Image, stype_String]'
        pixels = image.pixels
        stype = stype.get_string_value()
        if stype == 'Real':
            pixels = skimage.img_as_float(pixels)
        elif stype == 'Byte':
            pixels = skimage.img_as_ubyte(pixels)
        elif stype == 'Bit16':
            pixels = skimage.img_as_uint(pixels)
        elif stype == 'Bit':
            pixels = pixels.as_dtype(numpy.bool)
        else:
            return evaluation.error('ImageData', 'pixelfmt', stype);
        return from_python(pixels.tolist())


class ImageTake(Builtin):
    def apply(self, image, n, evaluation):
        'ImageTake[image_Image, n_Integer]'
        return Image(image.pixels[:int(n.to_python())], image.color_space)


class PixelValue(Builtin):
    def apply(self, image, x, y, evaluation):
        'PixelValue[image_Image, {x_?RealNumberQ, y_?RealNumberQ}]'
        return Real(image.pixels[int(y.to_python() - 1), int(x.to_python() - 1)])


class PixelValuePositions(Builtin):
    def apply(self, image, val, evaluation):
        'PixelValuePositions[image_Image, val_?RealNumberQ]'
        rows, cols = numpy.where(skimage.img_as_float(image.pixels) == float(val.to_python()))
        p = numpy.dstack((cols, rows)) + numpy.array([1, 1])
        return from_python(p.tolist())

# image attribute queries

class ImageDimensions(Builtin):
    def apply(self, image, evaluation):
        'ImageDimensions[image_Image]'
        return Expression('List', *image.dimensions())

class ImageAspectRatio(Builtin):
    def apply(self, image, evaluation):
        'ImageAspectRatio[image_Image]'
        dim = image.dimensions()
        return Real(dim[1] / float(dim[0]))

class ImageChannels(Builtin):
    def apply(self, image, evaluation):
        'ImageChannels[image_Image]'
        return Integer(image.channels())

class ImageType(Builtin):
    def apply(self, image, evaluation):
        'ImageType[image_Image]'
        return String(image.storage_type())

class BinaryImageQ(Test):
    def apply(self, image, evaluation):
        'BinaryImageQ[image_Image]'
        return Symbol('True') if image.storage_type() == 'Bit' else Symbol('False')

# Image core classes

class ImageCreate(Builtin):
    messages = {
        'noskimage': 'image creation needs scikit-image in order to work.'
    }

    def apply(self, array, evaluation):
        '''ImageCreate[array_?MatrixQ]'''
        if not _enabled:
            return evaluation.message('ImageCreate', 'noskimage')
        else:
            pixels = numpy.array(array.to_python(), dtype='float64')
            shape = pixels.shape
            is_rgb = (len(shape) == 3 and shape[2] == 3)
            if len(shape) == 2 or (len(shape) == 3 and shape[2] in (1, 3)):
                return Image(pixels.clip(0, 1), 'RGB' if is_rgb else 'Grayscale')
            else:
                return Expression('Image', array)


class ImageBox(BoxConstruct):
    def boxes_to_text(self, leaves, **options):
        return '-Image-'

    def boxes_to_xml(self, leaves, **options):
        # see https://tools.ietf.org/html/rfc2397
        img = '<img src="data:image/png;base64,%s" />' % (leaves[0].get_string_value())
        return '</math><mtable>%s</mtable><math>' % img

    def boxes_to_tex(self, leaves, **options):
        return '-Image-'


class Image(Atom):
    def __init__(self, pixels, color_space, **kwargs):
        super(Image, self).__init__(**kwargs)
        self.pixels = pixels
        self.color_space = color_space

    def pil(self):
        return PIL.Image.fromarray(self.pixels)

    def color_convert(self, to_color_space):
        if to_color_space == self.color_space:
            return self
        else:
            conversion = '%s2%s' % (self.color_space, to_color_space)
            if conversion in _color_space_conversions:
                return Image(_color_space_conversions[conversion](self.pixels), to_color_space)
            else:
                raise ValueError('cannot convert from color space %s to %s' % (self.color_space, to_color_space))

    def grayscale(self):
        return self.color_convert('Grayscale')

    def make_boxes(self, form):
        try:
            if self.color_space == 'Grayscale':
                pixels = self.pixels
            else:
                pixels = self.color_convert('RGB').pixels

            if pixels.dtype == numpy.bool:
                pixels = skimage.img_as_ubyte(pixels)

            shape = pixels.shape

            width = shape[1]
            height = shape[0]

            # if the image is very small, scale it up using nearest neighbour.
            min_size = 128
            if width < min_size and height < min_size:
                scale = min_size / max(width, height)
                pixels = skimage.transform.resize(pixels, (int(scale * height), int(scale * width)), order=0)

            stream = BytesIO()
            skimage.io.imsave(stream, pixels, 'pil', format_str='png')
            stream.seek(0)
            contents = stream.read()
            stream.close()

            encoded = base64.b64encode(contents)
            if not six.PY2:
                encoded = encoded.decode('utf8')

            return Expression('ImageBox', String(encoded), Integer(width), Integer(height))
        except:
            return Symbol("$Failed")

    def __str__(self):
        return '-Image-'

    def do_copy(self):
        return Image(self.pixels)

    def default_format(self, evaluation, form):
        return '-Image-'

    def get_sort_key(self, pattern_sort=False):
        if pattern_sort:
            return super(Image, self).get_sort_key(True)
        else:
            return hash(self)

    def same(self, other):
        return isinstance(other, Image) and numpy.array_equal(self.pixels, other.pixels)

    def to_sympy(self, **kwargs):
        return '-Image-'

    def to_python(self, *args, **kwargs):
        return self.pixels

    def __hash__(self):
        return hash(("Image", self.pixels.tobytes()))

    def dimensions(self):
        shape = self.pixels.shape
        return (shape[1], shape[0])

    def channels(self):
        shape = self.pixels.shape
        if len(shape) < 3:
            return 1
        else:
            return shape[2]

    def storage_type(self):
        dtype = self.pixels.dtype
        if dtype in (numpy.float32, numpy.float64):
            return 'Real'
        elif dtype == numpy.uint32:
            return 'Bit32'
        elif dtype == numpy.uint16:
            return 'Bit16'
        elif dtype == numpy.uint8:
            return 'Byte'
        elif dtype == numpy.bool:
            return 'Bit'
        else:
            return str(dtype)