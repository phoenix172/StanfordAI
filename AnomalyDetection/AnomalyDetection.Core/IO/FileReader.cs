
namespace AnomalyDetection.Core.IO;

public interface IFileReader
{
    Task<string[]> ReadFileLines(string dataPath);
}

public class FileReader : IFileReader
{
    private readonly string? _contentRootPath;
    private readonly HttpClient? _client;

    public FileReader(HttpClient client)
    {
        _client = client;
    }
    public FileReader()
        : this(Directory.GetCurrentDirectory())
    {
    }

    public FileReader(string contentRootPath)
    {
        _contentRootPath = contentRootPath;
    }

    public async Task<string[]> ReadFileLines(string dataPath)
    {
        string data;
        if (_client != null)
        {
            data = await _client.GetStringAsync(dataPath);
        }
        else
        {
            var filePath = RelativeToWebRoot(dataPath);
            data = await File.ReadAllTextAsync(filePath);
        }

        var split = data.Split(Environment.NewLine.ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
        return split;
    }

    private string RelativeToWebRoot(string dataPath)
    {
        return Path.Combine(_contentRootPath ?? string.Empty, dataPath);
    }
}