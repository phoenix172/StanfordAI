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
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
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
        private RegressionModel _model;

        MathNet.Numerics.LinearAlgebra.Vector<double> inputY;
        Matrix<double> inputX;

        public MainWindow()
        {
            _model = new RegressionModel();
            InitializeComponent();
        }

        private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            using var streamReader = new StreamReader("car_prices.csv");
            using var csvReader = new CsvHelper.CsvReader(streamReader, CultureInfo.InvariantCulture, false);
            var records = csvReader.GetRecords<CarPriceRecord>().ToList();

            var doors = records.Select(x => x.DoorNumber).Distinct();
            var cylinders = records.Select(x => x.CylinderNumber).Distinct();
            
            inputX = Matrix<double>.Build.DenseOfRowVectors(records.Select(x => MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfArray(new [] { x.CylindersCount, x.CityMpg, x.HorsePower, x.WheelBase })).ToArray());
            inputY = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(records.Select(x => x.Price));

            ShowScatter(inputX, inputY.ToArray(), xLabel: "Cylinders",yLabel:"Price");
            ShowScatter(inputX, inputY.ToArray(), 1, xLabel: "CityMpg",yLabel:"Price");
            ShowScatter(inputX, inputY.ToArray(), 2, xLabel: "HorsePower",yLabel:"Price");
            ShowScatter(inputX, inputY.ToArray(), 3, xLabel: "WheelBase",yLabel:"Price");
        }

        private void ShowScatter(Matrix<double> inputX, double[] inputY, int columnIndex=0, string xLabel="", string yLabel="")
        {
            var plot = new WpfPlot();
            var xNormalized = Normalize(inputX.Column(columnIndex).ToArray());
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
        

        private ObservableCollection<double> CostHistory { get; } = new ObservableCollection<double>();
        private void GradientDescent_Click(object sender, RoutedEventArgs e)
        {
            var costs = _model.Fit(inputX, inputY).Select((x,i) =>
            {
                if(i%10000==0)
                    Debug.WriteLine($"iteration {i}: cost {x}" );
                return x;
            }).ToList();

            costHistoryPlot.Plot.Frameless(false);
            costHistoryPlot.Plot.AddSignalXY(Enumerable.Range(0, costs.Count).Select(x=>(double)x).ToArray(), costs.ToArray());
            //parametersDisplay.Content = string.Join(",", output) + $" Cost: {cost}";
            costHistoryPlot.Refresh();

            parametersDisplay.Content = ("Model converged at parameters: " + string.Join(",", _model.Weight) +
                            $",{_model.Bias} with Cost: {costs.Last()}");
        }

        private void PredictClick(object sender, RoutedEventArgs e)
        {
            var input = new[]
                { double.Parse(tbCylinders.Text), double.Parse(tbCityMpg.Text), double.Parse(tbHorsePower.Text), double.Parse(tbWheelBase.Text) };
            double result =
                _model.ComputePrediction(MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfArray(input));
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
