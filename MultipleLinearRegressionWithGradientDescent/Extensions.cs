using System;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using ScottPlot;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Runtime.CompilerServices;
using OxyPlot;

namespace MultipleLinearRegressionWithGradientDescent;

public static class Extensions
{
    public static async Task<double[]> FitAndPlot(this RegressionModel model, WpfPlot plot,
        Matrix<double> trainingInput, Vector<double> trainingOutput)
    {
        var cost = await Task.Run(() =>
        {
            var costList = plot.Plot.AddScatterList(markerSize: 1.5f, lineStyle: LineStyle.Solid,
                lineWidth: 0.5f, markerShape: MarkerShape.none);
            int chunkSize = 1000;
            var costs = model.Fit(trainingInput, trainingOutput).Chunk(chunkSize)
                .Select((x, i) =>
                {
                    var costs = x.ToList();
                    Debug.WriteLine($"iteration {i}: cost {x.Last()}");
                    var xs = Enumerable.Range(chunkSize * i, costs.Count).Select(y => (double)y).ToArray();
                    var ys = costs.ToArray();

                    var finite = xs.Zip(ys).Where(x => double.IsFinite(x.Second));
                    costList.AddRange(finite.Select(x=>x.First).ToArray(),finite.Select(x=>x.Second).ToArray());
                    App.Current.Dispatcher.Invoke(() => plot.RenderRequest());
                    return x;
                }).ToList();
            return costs.Last();
        });
        return cost;
    }

    public static T1[,] Grid<T, T1>(this IEnumerable<T> input, Func<T,T,T1> valueFunc)
    {
        var data = input.ToList();
        var result = new T1[data.Count,data.Count];
        for (int i = 0; i < data.Count; i++)
        {
            for (int j = 0; j < data.Count; j++)
            {
                result[i,j] = valueFunc(data[i], data[j]);
            }
        }

        return result;
    }

    public static void PlotBoundary(this LogisticalRegressionModel model, WpfPlot plot)
    {
        var rowMajorOriginal = model.OriginalTrainingInput.ToRowMajorArray();
        var min = rowMajorOriginal.Min();
        var max = rowMajorOriginal.Max();

        min -= Math.Abs(min) * 0.25;
        max += Math.Abs(max) * 0.25;

        var range = RangeStep(min, max ,100);

        double minZ = double.MaxValue;
        double maxZ = double.MinValue;
        var valuesGrid = range
            .Grid((x, y) =>
            {
                var z = model.Predict(Vector<double>.Build.Dense(new[] { x, y }));
                minZ = Math.Min(z, minZ);
                maxZ = Math.Max(z, maxZ);
                return z;
            });

        void PlotNormalLine(double x1, double y1, double x2, double y2, double z)
        {
            var a = model.NormalizedInput.NormalizeRow(model.FeatureMap(Vector<double>.Build.Dense(new[] { x1, y1 })));
            var b = model.NormalizedInput.NormalizeRow(model.FeatureMap(Vector<double>.Build.Dense(new[] { x2, y2 })));
            plot.Plot.AddLine(a[0], a[1], b[0], b[1], Color.Blue);
        }

        Conrec.Contour(valuesGrid, range.ToArray(), range.ToArray(), new[] { 0.5 }, PlotNormalLine);
        plot.Refresh();
    }

    public static double[] RangeStep(double start, double end, int steps)
    {
        double stepSize = Math.Abs(start - end) / steps;
        return Enumerable.Range(0, steps).Select(x => start + x * stepSize).ToArray();
    }
}