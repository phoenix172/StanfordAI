using MathNet.Numerics.LinearAlgebra;
using ScottPlot;
using ScottPlot.Plottable;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using OxyPlot;
using System.Collections.Generic;

namespace NeuralNetworks;

public static class Extensions
{

    public static Matrix<double> ColumnExpand(this Vector<double> input, int count)
        => Matrix<double>.Build.DenseOfColumns(Enumerable.Repeat(input, count));
    public static Matrix<double> RowExpand(this Vector<double> input, int count)
        => Matrix<double>.Build.DenseOfRows(Enumerable.Repeat(input, count));


    public static void PlotBoundary(this NeuralNetworkModel model, ScatterPlotList<double> plotList)
    {
        plotList.Clear();

        var rowMajorOriginal = model.TrainingInput.ToRowMajorArray();
        var min = rowMajorOriginal.Min();
        var max = rowMajorOriginal.Max();

        min -= Math.Abs(min) * 0.25;
        max += Math.Abs(max) * 0.25;

        var range = RangeStep(min, max, 200);

        double minZ = double.MaxValue;
        double maxZ = double.MinValue;

        double[][] testData = range
            .Grid((x, y) => (x, y)).OfType<(double, double)>().Select(x => new[] { x.Item1, x.Item2 }).ToArray();
        var testInput = Matrix<double>.Build.DenseOfRowArrays(testData);
        var predictions = NeuralNetworkModel.Softmax(model.Predict(testInput)).EnumerateRows()
            .Select(x => (double)x.MaximumIndex());

        Conrec.Contour(Make2DArray(predictions.ToArray(), range.Length, range.Length), range.ToArray(), range.ToArray(), new[] { 0.5 }, PlotLine(plotList));
    }


    public static double[] RangeStep(double start, double end, int steps)
    {
        double stepSize = Math.Abs(start - end) / steps;
        return Enumerable.Range(0, steps).Select(x => start + x * stepSize).ToArray();
    }


    public static T1[,] Grid<T, T1>(this IEnumerable<T> input, Func<T, T, T1> valueFunc)
    {
        var data = input.ToList();
        var result = new T1[data.Count, data.Count];
        for (int i = 0; i < data.Count; i++)
        {
            for (int j = 0; j < data.Count; j++)
            {
                result[i, j] = valueFunc(data[i], data[j]);
            }
        }

        return result;
    }

    private static T[,] Make2DArray<T>(T[] input, int height, int width)
    {
        T[,] output = new T[height, width];
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                output[i, j] = input[i * width + j];
            }
        }
        return output;
    }



    public static Conrec.RendererDelegate PlotLine(ScatterPlotList<double> plot) =>
        (x1, y1, x2, y2, z) =>
        {
            plot.Add(x1, y1);
            plot.Add(x2, y2);
        };

    public static async Task<double[]> FitAndPlot(this NeuralNetworkModel model, WpfPlot costPlot,
        Matrix<double> trainingInput, Vector<double> trainingOutput, int chunkSize = 1000, WpfPlot? dataPlot = null, int epochs = 1000)
    {
        var costList = costPlot.Plot.AddScatterList(markerSize: 1.5f, lineStyle: LineStyle.Solid,
            lineWidth: 0.5f, markerShape: MarkerShape.none);
        ScatterPlotList<double>? functionPlot = null;

        if (dataPlot != null)
        {
            functionPlot = dataPlot.Plot.AddScatterList<double>(Color.SaddleBrown,lineStyle:LineStyle.None);
            model.PlotBoundary(functionPlot);
        }

        double[] PlotChunk(double[] x, int i)
        {
            var costs = x.ToList();
            Debug.WriteLine($"iteration {i}: cost {x.Last()}");

            var xs = Enumerable.Range(chunkSize * i, costs.Count).Select(y => (double)y).ToArray();
            var ys = costs.Select(x=>Math.Exp(Math.Exp(Math.Exp(x)))).ToArray();

            var finite = xs.Zip(ys).Where(x => double.IsFinite(x.Second) && !double.IsNaN(x.Second));
            costList.AddRange(finite.Select(x => x.First).ToArray(), finite.Select(x => x.Second).ToArray());

            if (dataPlot != null)
                model.PlotBoundary(functionPlot);

            App.Current.Dispatcher.Invoke(() =>
            {
                costPlot.RenderRequest();
                dataPlot?.RenderRequest();
            });


            return x;
        }

        var cost = await Task.Run(() =>
        {
            var costs = model.Fit(trainingInput, trainingOutput, epochs, 128).Chunk(chunkSize)
                .Select(PlotChunk).ToList();
            return costs.Last();
        });
        return cost;
    }

}