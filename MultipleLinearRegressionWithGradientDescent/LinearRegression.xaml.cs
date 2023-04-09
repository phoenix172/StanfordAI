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
using ScottPlot.Plottable;
using ScottPlot.Renderable;
using Vector = System.Numerics.Vector;

namespace MultipleLinearRegressionWithGradientDescent
{
    /// <summary>
    /// Interaction logic for LinearRegression.xaml
    /// </summary>
    public partial class LinearRegression : Window
    {
        private readonly RegressionModel _model;
        MathNet.Numerics.LinearAlgebra.Vector<double> _inputY;
        Matrix<double> _inputX;

        public List<CarPriceRecord> Records { get; private set; }

        public LinearRegression()
        {
            _model = new RegressionModel();

            using var streamReader = new StreamReader("car_prices.csv");
            using var csvReader = new CsvHelper.CsvReader(streamReader, CultureInfo.InvariantCulture, false);
            Records = csvReader.GetRecords<CarPriceRecord>().ToList();
            
            _inputX = Matrix<double>.Build.DenseOfRowVectors(Records.Select(x => MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfArray(new[] { x.CylindersCount, x.CityMpg, x.HorsePower, x.WheelBase })).ToArray());
            _inputY = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(Records.Select(x => x.Price));

            InitializeComponent();
        }

        private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
        }

        private void ShowScatter(double[] inputY, int columnIndex = 0, string xLabel = "", string yLabel = "")
        {
            var plot = new WpfPlot();
            var xNormalized = _model.TrainingInput.Column(columnIndex).ToArray();
            plot.Plot.AddScatter(xNormalized, inputY.Select(x => (double)x).ToArray(),
                lineStyle: LineStyle.None);
            plot.Plot.XLabel(xLabel);
            plot.Plot.YLabel(yLabel);
            plots.Children.Add(plot);
            plot.Refresh();
        }

        private async void GradientDescent_Click(object sender, RoutedEventArgs e)
        {
            _model.LearningRate = 1;
            _model.TrainingThreshold = 3E2;
            var cost = await _model.FitAndPlot(costHistoryPlot, _inputX, _inputY);

            parametersDisplay.Content = ("Model converged at parameters: " + string.Join(",", _model.Weight) +
                                         $",{_model.Bias} with Cost: {cost}");


            ShowScatter(_inputY.ToArray(), xLabel: "Cylinders", yLabel: "Price");
            ShowScatter(_inputY.ToArray(), 1, xLabel: "CityMpg", yLabel: "Price");
            ShowScatter(_inputY.ToArray(), 2, xLabel: "HorsePower", yLabel: "Price");
            ShowScatter(_inputY.ToArray(), 3, xLabel: "WheelBase", yLabel: "Price");

        }

        private void PredictClick(object sender, RoutedEventArgs e)
        {
            var input = new[]
                { double.Parse(tbCylinders.Text), double.Parse(tbCityMpg.Text), double.Parse(tbHorsePower.Text), double.Parse(tbWheelBase.Text) };
            double result =
                _model.Predict(MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfArray(input));
            lbPrice.Content = result;
        }

        private void LogisticalRegression_Click(object sender, RoutedEventArgs e)
        {
            LogisticalLinearRegression a = new LogisticalLinearRegression();
            a.Show();
        }
    }

    public record CarPriceRecord(string DoorNumber, double CurbWeight, string CylinderNumber, double EngineSize,
        string FuelType, double WheelBase, double HorsePower, double CityMpg, double HighWayMpg, double Price)
    {
        public int DoorsCount => DoorNumber switch
        {
            "two" => 2,
            "four" => 4,
            _ => -1
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
