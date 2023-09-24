using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.Input;
using KMeansImageCompression.Data;
using KMeansImageCompression.Model;

namespace KMeansImageCompression;

public partial class MainViewModel
{
    public MainViewModel()
        : this(TestData.OriginalImage, TestData.OriginalImage)
    {
    }

    public MainViewModel(BitmapImage originalImage, BitmapImage compressedImage)
    {
        OriginalImage = originalImage;
        CompressedImage = compressedImage;
    }

    public BitmapImage OriginalImage { get; }
    public BitmapImage CompressedImage { get; }

    [RelayCommand]
    public void CompressImage()
    {
            new ImageCompressorService().Load();

    }

    
}