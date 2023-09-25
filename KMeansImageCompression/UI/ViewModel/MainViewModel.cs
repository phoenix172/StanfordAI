using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;

namespace KMeansImageCompression;

[ObservableObject]
public partial class MainViewModel
{
    [ObservableProperty] private BitmapSource _compressedImage;
    [ObservableProperty] private int _iterations = 24;
    [ObservableProperty] private int _colors = 16;


    public MainViewModel()
        : this(TestData.OriginalImage.To24BitFormat())
    {
    }

    public MainViewModel(BitmapSource image)
    {
        OriginalImage = image;
        _compressedImage = image;
    }

    public BitmapSource OriginalImage { get; }

    [RelayCommand(AllowConcurrentExecutions = false)]
    public async Task CompressImage()
    {
        var a = new CompressedImage(OriginalImage,new()
        {
            IterationsCount = Iterations,
            TargetColorsCount = Colors
        });
        CompressedImage = await a.CompressImageAsync();

        CompressedImage.SaveToFile("compressed.png");
    }
}