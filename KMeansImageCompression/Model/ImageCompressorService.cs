using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using KMeansImageCompression.Data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NumSharp;

namespace KMeansImageCompression.Model;

public class ImageCompressorService
{

    public ImageCompressorService()
    {

    }

    public BitmapSource CompressImage(BitmapSource originalImage)
    {
        var pixels = originalImage.GetPixels();
        Matrix<double> pixelMatrix = pixels.ToMatrix();

        var centroids = CreateCentroidMatrix(pixelMatrix, 16);
        Vector<double>? closestCentroidIndices = null;
        Matrix<double>? kMeans = null;
        for (int i = 0; i < 24; i++)
        { 
            //Vector<double>
                closestCentroidIndices = FindClosestCentroids(pixelMatrix, centroids); //new color index for each pixel

            //Matrix<double> 
                kMeans = ComputeCentroids(pixelMatrix, closestCentroidIndices); //new colors\
                centroids = kMeans;
        }

    var compressedBitmapMatrix = Matrix<double>.Build.DenseOfRowVectors
        (
        closestCentroidIndices
                .EnumerateIndexed()
                .Select(x => kMeans.Row((int)x.Item2))
        );

        var compressedImage = compressedBitmapMatrix.ToBitmapImage(originalImage);

        return compressedImage;
    }

    public Matrix<double> CreateCentroidMatrix(Matrix<double> pixels, int centroidsCount)
    {
        Random random = new Random(DateTime.Now.ToFileTimeUtc().GetHashCode());
        var randomPixelIndices =
            Enumerable.Range(0, centroidsCount).Select(x => (int)random.NextInt64(0, pixels.RowCount - 1));

        var randomPixels = Matrix<double>.Build.DenseOfRowVectors(randomPixelIndices.Select(pixels.Row));

        return randomPixels;
    }

    public Vector<double> FindClosestCentroids(Matrix<double> pixelMatrix, Matrix<double> centroidMatrix)
    {
        //Matrix<Centroid(K) by Pixel(n=s*s)>
        Matrix<double> centroidDistanceMatrix = Matrix<double>.Build.DenseOfRowVectors
        (
        centroidMatrix
                .EnumerateRows()
                .Select(centroid => Extensions.DistanceToCentroid(pixelMatrix, centroid))
        );

        //Result Vector<int> : index of closest centroid
        var closestCentroids = Vector<double>.Build.DenseOfEnumerable
        (
            centroidDistanceMatrix
                .EnumerateColumns()
                .Select(pixelDistancesColumn => (double)pixelDistancesColumn.MinimumIndex())
        );

        return closestCentroids;
    }

    public Matrix<double> ComputeCentroids(Matrix<double> pixels, Vector<double> closestCentroids)
    {
        var kMeansEnumerable = pixels.EnumerateRowsIndexed()
            .GroupBy(pixel => closestCentroids[pixel.Item1], x => x.Item2)
            .Select(Matrix<double>.Build.DenseOfRowVectors)
            .Select(cluster => cluster.ColumnSums().Divide(cluster.RowCount));

        var kMeans = Matrix<double>.Build.DenseOfRowVectors(kMeansEnumerable);

        return kMeans;
    }

    public void Load()
    {

    }
}