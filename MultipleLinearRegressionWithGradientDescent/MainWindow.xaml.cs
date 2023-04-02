using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ScottPlot;
using ScottPlot.Renderable;
using Vector = System.Numerics.Vector;

namespace MultipleLinearRegressionWithGradientDescent
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        double[] inputY;
        Vector<double>[] inputX;

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            using var streamReader = new StreamReader("car_prices.csv");
            using var csvReader = new CsvHelper.CsvReader(streamReader, CultureInfo.InvariantCulture, false);
            var records = csvReader.GetRecords<CarPriceRecord>().ToList();

            var doors = records.Select(x => x.DoorNumber).Distinct();
            var cylinders = records.Select(x => x.CylinderNumber).Distinct();

            new Vector<double>(new double[]{1, 2, 3,0});
            inputX = records.Select(x => new Vector<double>(Normalize(new double[] { x.CylindersCount, x.CityMpg, x.HorsePower, x.WheelBase }))).ToArray();
            inputY = records.Select(x => x.Price).ToArray();

            ShowScatter(inputX, inputY, xLabel: "Cylinders",yLabel:"Price");
            ShowScatter(inputX, inputY, 1, xLabel: "CityMpg",yLabel:"Price");
            ShowScatter(inputX, inputY, 2, xLabel: "HorsePower",yLabel:"Price");
            ShowScatter(inputX, inputY, 3, xLabel: "WheelBase",yLabel:"Price");
        }

        private void ShowScatter(Vector<double>[] inputX, double[] inputY, int columnIndex=0, string xLabel="", string yLabel="")
        {
            var plot = new WpfPlot();
            var xNormalized = Normalize(inputX.Select(x => x[columnIndex]).ToArray());
            plot.Plot.AddScatter(xNormalized, inputY.Select(x => (double)x).ToArray(),
                lineStyle: LineStyle.None);
            plot.Plot.XLabel(xLabel);
            plot.Plot.YLabel(yLabel);
            plots.Children.Add(plot);
            plot.Refresh();
        }

        private double[] Normalize(double[] input)
        {
            double deviation = StandardDeviation(input);
            double mean = input.Average();
            return input.Select(x => (x - mean) / deviation).ToArray();
        }

        private double StandardDeviation(double[] input)
        {
            double mean = input.Average();
            double deviation = input.Sum(x => Math.Pow(x - mean, 2));
            return deviation / input.Length;
        }

        private double ComputePrediction(Vector<double> input,Vector<double> weight, double constant)
        {
            return Vector.Dot(input, weight) + constant;
        }

        private double ComputeCost(Vector<double>[] inputData, double[] targets, Vector<double> weight, double constant)
        {
            double InstanceCost(Vector<double> x, int i) 
                => Math.Pow(ComputePrediction(x, weight, constant) - targets[i], 2);

            return inputData.Select(InstanceCost).Sum();
        }

        private double[] ComputeGradient(Vector<double>[] inputData, double[] targets, Vector<double> weight, double constant)
        {
            double[] predictions = inputData.Select(x => ComputePrediction(x, weight, constant)).ToArray();

            Debug.Assert(predictions.Length != 0);

            double GradientComponent(int featureIndex)
            {
                double[] featureValues = inputData.Select(x => x[featureIndex]).ToArray();
                return predictions.Select((x, i) => (x - targets[i]) * featureValues[i]).Sum() / predictions.Length;
            }

            var constantGradientComponent = predictions.Select((x, i) => x - targets[i]).Sum() / predictions.Length;

            return Enumerable.Range(0, 4).Select(GradientComponent).Append(constantGradientComponent).ToArray();
        }

        private Vector<double>? weights;
        private double? constant;
        private double learningRate = 0.1;

        private ObservableCollection<double> CostHistory { get; } = new ObservableCollection<double>();
        private void GradientDescent_Click(object sender, RoutedEventArgs e)
        {
            double lastCost = GradientStep();
            double cost = GradientStep();
            double epsilon = 3E3;

            List<double> costs = new
                List<double>();
            while (Math.Abs(cost-lastCost) > epsilon)
            {
                lastCost = cost;
                cost = GradientStep();
                costs.Add(cost);
            }

            costHistoryPlot.Plot.Frameless(false);
            costHistoryPlot.Plot.AddSignalXY(Enumerable.Range(0, costs.Count).Select(x=>(double)x).ToArray(), costs.ToArray());
            //parametersDisplay.Content = string.Join(",", output) + $" Cost: {cost}";
            costHistoryPlot.Refresh();

            parametersDisplay.Content = ("Model converged at parameters: " + string.Join(",", weights) +
                            $",{constant} with Cost: {cost}");
        }

        private double GradientStep()
        {
            constant ??= 0;
            weights ??= new Vector<double>(new double[] { 0, 0, 0, 0 });
            var output = ComputeGradient(inputX, inputY, weights.Value, constant.Value);
            weights = weights - learningRate * new Vector<double>(output[..4]);
            constant -= learningRate * output.Last();
            var cost = ComputeCost(inputX, inputY, weights.Value, constant.Value);
            //Debug.WriteLine(string.Join(",", output) + $" Cost: {cost}");
            return cost;
        }

        private void PredictClick(object sender, RoutedEventArgs e)
        {
            var input = new[]
                { double.Parse(tbCylinders.Text), double.Parse(tbCityMpg.Text), double.Parse(tbHorsePower.Text), double.Parse(tbWheelBase.Text) };
            double result = ComputePrediction(new Vector<double>(Normalize(input)), weights.Value, constant.Value);
            lbPrice.Content = result;
        }
    }

    record CarPriceRecord(string DoorNumber, double CurbWeight, string CylinderNumber, double EngineSize,
        string FuelType, double WheelBase, double HorsePower, double CityMpg, double HighWayMpg, double Price)
    {
        public int DoorsCount => DoorNumber switch
        {
            "two" => 2,
            "four" => 4,
            _=>-1
        };

        public int CylindersCount => CylinderNumber switch
        {
            "two" => 2,
            "three" => 3,
            "four" => 4,
            "five" => 5,
            "six" => 6,
            "eight" => 8,
            "ten" => 10,
            "twelve" => 12,
            _ => -1
        };
    };
}
