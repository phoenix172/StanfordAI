using System;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace KMeansImageCompression.Model;

public class ImageCompressorService
{
    public void CompressImage(BitmapImage originalImage)
    {
        var pixels = originalImage.GetPixels();
    }

    public void FindClosestCentroids(PixelColor[,] pixels, PixelColor[] centroids)
    {
        var flatPixels = pixels.Cast<PixelColor>().ToArray();
        var pixelCentroids = from pixel in flatPixels
            from centroid in centroids
            select (pixel, centroid);
        //pixelCentroids.Select(x => DistanceToCentroid(x.pixel, x.centroid));
    }

    private Vector<double> DistanceToCentroid(Matrix<double> pixels, Matrix<double> centroid)
    {
        return (pixels - centroid).PointwiseAbs().RowSums();

    }
}