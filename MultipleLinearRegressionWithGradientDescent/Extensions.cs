using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using ScottPlot;
using System.Threading.Tasks;

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
                    costList.AddRange(Enumerable.Range(chunkSize * i, costs.Count).Select(y => (double)y).ToArray(),
                        costs.ToArray());
                    App.Current.Dispatcher.Invoke(() => plot.RenderRequest());
                    return x;
                }).ToList();
            return costs.Last();
        });
        return cost;
    }
}