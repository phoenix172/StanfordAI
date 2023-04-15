using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ScottPlot;
using Color = System.Drawing.Color;

namespace MultipleLinearRegressionWithGradientDescent
{
    public record DataPoint(double X, double Y);

    /// <summary>
    /// Interaction logic for GenericLinearRegression.xaml
    /// </summary>
    public partial class GenericLinearRegression : Window
    {
        private readonly Matrix<double> _inputX;
        private readonly Vector<double> _inputY;
        public ObservableCollection<DataPoint> Records { get; set; }
        RegressionModel _model;


        public GenericLinearRegression(int a)
        {
            InitializeComponent();
            string dataPath;
            _model = new RegressionModel();

            if (a == 0)
            {
                dataPath = "Data/secondary.csv";
                _model.FeatureMap = RegressionModel.MapFeatureDegree(12);
                _model.BatchSize = 16;
                _model.RegularizationTerm = 0.2;
                _model.LearningRate = 1e-1;
                _model.TrainingThreshold = 1e-32;
            }
            else
            {
                dataPath = "Data/primary.csv";
                _model.FeatureMap = RegressionModel.MapFeatureDegree(12);
                _model.BatchSize = 16;
                _model.RegularizationTerm = 0.2;

                _model.LearningRate = 1e-1;
                _model.TrainingThreshold = 1e-32;
            }

            using var streamReader = new StreamReader(dataPath);
            using var csvReader = new CsvHelper.CsvReader(streamReader, CultureInfo.InvariantCulture, false);
            Records = new ObservableCollection<DataPoint>(csvReader.GetRecords<DataPoint>().OrderBy(x => x.X));

            _inputX = Matrix<double>.Build.DenseOfRowVectors(Records.Select(x => Vector<double>.Build.DenseOfArray(new[] { x.X })).ToArray());
            _inputY = Vector<double>.Build.DenseOfEnumerable(Records.Select(x => x.Y));



        }

        private void GenericLinearRegression_OnLoaded(object sender, RoutedEventArgs e)
        {
            DataPlot.Plot.AddSignalXY(_inputX.Column(0).ToArray(), _inputY.ToArray());
            DataPlot.Refresh();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            double[] cost = await _model.FitAndPlot(CostPlot, _inputX, _inputY, 10000, DataPlot);


            Debug.WriteLine(_model.GetModelEquation());
            Debug.WriteLine(_model.GetNormalizationEquation(0));
            Debug.WriteLine(_model.GetModelExpression());
        }

    }
}
