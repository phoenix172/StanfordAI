namespace AnomalyDetection.Client.Business;

public static class CrossPlatform
{
    private static readonly char[] PathSplitCharacters = new char[] { '/', '\\' };

    public static string PathCombine(string basePath, params string[] additional)
    {
        var splits = additional.Select(s => s.Split(PathSplitCharacters)).ToArray();
        var totalLength = splits.Sum(arr => arr.Length);
        var segments = new string[totalLength + 1];
        segments[0] = basePath;
        var i = 0;
        foreach (var split in splits)
        {
            foreach (var value in split)
            {
                i++;
                segments[i] = value;
            }
        }
        return Path.Combine(segments).TrimStart('/');
    }
}