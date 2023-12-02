namespace AnomalyDetection.Client.Business;

public class FileReader
{
    private readonly HttpClient? _client;

    public FileReader(HttpClient? client = null)
    {
        _client = client;
    }

    public async Task<string[]> ReadFileLines(string dataPath)
    {
        string data = string.Empty;
        if (_client != null)
        {
            data = await _client.GetStringAsync(dataPath);
        }
        else
        {
            data = await File.ReadAllTextAsync(dataPath);
        }

        return data.Split(Environment.NewLine);
    }
}