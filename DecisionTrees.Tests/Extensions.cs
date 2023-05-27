namespace DecisionTrees;

public static class Extensions
{
    public static T[][] ToJagged<T>(this T[,] obj2D)
    {
        T[] obj1D = obj2D.Cast<T>().ToArray();

        int j = 0;
        T[][] jagged = obj1D
            .GroupBy(_ => j++ / obj2D.GetLength(1))
            .Select(y => y.ToArray()).ToArray();

        return jagged;
    }

    public static IEnumerable<int> IndicesOf<T>(this T[] array, IEnumerable<T> items)
    {
        return items.Select(x => Array.IndexOf(array, x));
    }
}